# This code was originally adapted from this repository: https://github.com/YueJiang-nj/CVPR2020-SDFDiff
# Please see the original repository for more details on the raymarching computation

from __future__ import print_function
import torch
import math
import torchvision
from torch.autograd import Variable
import time
import sys, os
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import pickle
import math
import argparse
import copy
import numpy as np
from tqdm import tqdm
from scipy.constants import c
from scipy.signal import find_peaks

sys.path.append('..')
sys.path.append('../..')
from utils.generic_loader import *
import ray_tracing

UNDEF_VAL = -50 
USE_RAY_TRACING = True

def _read_sdf_file(obj_id, name, is_los, ext):
    """ 
    Read inputs from the sdf file and reformat as needed
    Parameters:
        - obj_id (str): ID of object
        - name (str): Name of object
        -ext (str): Extra string appended to filename to load 
    """
    loader = GenericLoader(obj_id, name, is_sim=False, is_los=is_los, exp_num='2') # Note: Only exp 2 is supported at this time
    data = loader.load_sdf_file(radar_type='77_ghz', angles=None, ext=ext)

    sdf_flat = torch.from_numpy(data['rsdf'])
    normals_flat = torch.from_numpy(data['normals']) 
    rx_locs = torch.from_numpy(data['rx_locs'])
    rx_dirs = torch.from_numpy(data['rx_dirs'])
    channels = torch.from_numpy(data['channels'])
    component_labels = torch.from_numpy(np.int32(data['component_labels']))
    valid_labels = torch.from_numpy(data['valid_labels'])
    bounding_box = data['bounding_box']
    wavelengths = torch.from_numpy(data['wavelengths'])
    voxel_size = data['voxel_size']
    num_samples = data['num_samples']
    coords = data['coords'] 
    sum_img = data['sum_img']

    # Normalize channels
    channels *= 0.00001/6 

    # change type for c++
    rx_dirs = rx_dirs.type(torch.float32)
    sdf_flat = sdf_flat.type(torch.float32)

    # Remove background from valid labels
    if 0 in valid_labels: # 0 is background
        valid_labels = valid_labels[valid_labels!=0]

    # Replace NAN with a pre-defined value (for easier use in C++)
    sdf_flat[torch.isnan(sdf_flat)] = UNDEF_VAL
    normals_flat[torch.isnan(normals_flat)] = UNDEF_VAL

    # Reformatt data for our use
    sdf = torch.unsqueeze(sdf_flat, -1)
    sdf = sdf.repeat(1,1,1,len(valid_labels))
    normals = torch.unsqueeze(normals_flat, -1)
    normals = normals.repeat(1,1,1,1,len(valid_labels))
    tau_bounds = torch.zeros(len(valid_labels), 2)
    for i, label in enumerate(valid_labels):
        sdf[:,:,:, i][component_labels!=label] = UNDEF_VAL
        normals[:,:,:,:,i][component_labels!=label,:] = UNDEF_VAL
        tau_bounds[i, 0] = torch.min(sdf_flat[torch.logical_and(component_labels==label, sdf_flat>UNDEF_VAL)])+0.015 # max SDF - max_sdf = 0
        tau_bounds[i, 1] = torch.max(sdf_flat[torch.logical_and(component_labels==label, sdf_flat>UNDEF_VAL)])-0.015 # min_sdf - min_sdf

    return sdf, normals, rx_locs, rx_dirs, channels, bounding_box, wavelengths, voxel_size, tau_bounds, loader, num_samples, coords, sum_img


def _compute_intersection_pos(grid, intersection_pos_rough, voxel_min_point, x,y,z, ray_direction, voxel_size, mask):
    """
    This function interpolates an SDF to find exactly where a ray intersects an object. Please see original repository for more details 
    """
    # Linear interpolate along x axis the eight values
    tx = (intersection_pos_rough[:,:,0] - voxel_min_point[:,:,0]) / voxel_size;

    c01 = (1 - tx) * grid[x,y,z].cuda() + tx * grid[x+1,y,z].cuda();
    c23 = (1 - tx) * grid[x,y+1,z].cuda() + tx * grid[x+1,y+1,z].cuda();
    c45 = (1 - tx) * grid[x,y,z+1].cuda() + tx * grid[x+1,y,z+1].cuda();
    c67 = (1 - tx) * grid[x,y+1,z+1].cuda() + tx * grid[x+1,y+1,z+1].cuda();
           
    # Linear interpolate along the y axis
    ty = (intersection_pos_rough[:,:,1] - voxel_min_point[:,:,1]) / voxel_size;
    c0 = (1 - ty) * c01 + ty * c23;
    c1 = (1 - ty) * c45 + ty * c67;

    # Return final value interpolated along z
    tz = (intersection_pos_rough[:,:,2] - voxel_min_point[:,:,2]) / voxel_size;

    sdf_value = (1 - tz) * c0 + tz * c1;

    return (intersection_pos_rough + ray_direction * sdf_value.view(width,height,1).repeat(1,1,3))\
                            + (1 - mask.view(width,height,1).repeat(1,1,3))
        

def _simulate_channels(bounding_box, bounding_box_lower, voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height,  width_fov, height_fov, grid, rx_loc, rx_dir, wavelengths, taus, tau_bounds, is_real_data, apply_tau_conversion=True, sdf_indices=None):
    """
    This function simulates mmWave channels from a given set of isosurfaces
    
    Parameters:
        - bounding_box: Defines a bounding box around the image 
        - bounding_box_lower:: The bottom corner of image, used as a reference point
        - voxel_size: size of a voxel in the SAR image/SDF
        - grid_res_x: Number of voxels in the x direction
        - grid_res_y: Number of voxels in the y direction
        - grid_res_z: Number of voxels in the z direction 
        - width: Number of rays to use in the horizontal dimension
        - height: Number of rays to use in the vertical dimension
        - width_fov: Total FoV over which to send rays in the horizontal dimension
        - height_fov: Total FoV over which to send rays in the vertical dimension
        - grid: SDF data
        - rx_loc: Locations of antenna for which to simulate channel
        - rx_dir: Direction of antenna (e.g., straight down in our case)
        - wavelengths: Wavelengths of FMCW chirp to simulate
        - taus: RSDF offsets to use to select isosurfaces to simulate
        - tau_bounds: Upper and lower bounds of RSDF offsets
        - is_real_data: Whether this data is real-world or simulation
        - apply_tau_conversion: Whether to clamp tau within its possible limits
        - sdf_indices: Which RSDF components to select isosurfaces from and simulate
    
    Returns a numpy array of simulated channels
    """
    # Prepare data
    dev = taus.device
    num_sdfs = grid.shape[-1]
    eye_x = rx_loc[0]
    eye_y = rx_loc[1]
    eye_z = rx_loc[2]
    rx_dir_x, rx_dir_y, rx_dir_z = rx_dir
    rx_pos_rep = rx_loc.repeat(width, height, 1)
    combined_channel = torch.zeros(wavelengths.shape, dtype=torch.complex64).to(device=dev)
    bounding_box_min_x, bounding_box_min_y, bounding_box_min_z \
        , bounding_box_max_x, bounding_box_max_y, bounding_box_max_z = bounding_box
    if sdf_indices is None:
        sdf_indices = torch.arange(num_sdfs)
    
    # Simulate channel for every applicable RSDF component 
    for sdf_idx in sdf_indices:
        # Select desired isosurface of this RSDF component
        if apply_tau_conversion:
            min_tau, max_tau = tau_bounds[sdf_idx]
            tau = torch.clip(taus[sdf_idx], min_tau, max_tau)
        grid_subset = grid[:,:,:,sdf_idx] - tau


        # Call ray tracing in cpp
        # This finds a rough estimate of points on the isosurface where rays from this antenna will intersect
        w_h_3 = torch.zeros(width, height, 3).cuda() # Sizes for cuda code 
        w_h = torch.zeros(width, height).cuda() # Sizes for cuda code 
        outputs = ray_tracing.ray_matching(w_h_3, w_h, grid_subset, width, height,  width_fov, height_fov, bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
                                        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                                        grid_res_x, grid_res_y, grid_res_z, \
                                        eye_x, \
                                        eye_y, \
                                        eye_z, \
                                        rx_dir_x, rx_dir_y, rx_dir_z
                                        )
        intersection_pos_rough = outputs[0] # Where the ray marching reached (WxHx3)
        voxel_min_point_index = outputs[1] # Index in SDF where it reached
        ray_direction = outputs[2] # Direction of each ray

        # Remove rays with no intersections
        mask = (voxel_min_point_index[:,:,0] != -1).type(Tensor).to(device=dev)
        if torch.sum(mask) == 0: continue # Dont continue if there are no intersections

        # Get the indices of the minimum point of the intersecting voxels
        x = voxel_min_point_index[:,:,0].type(torch.cuda.LongTensor)
        y = voxel_min_point_index[:,:,1].type(torch.cuda.LongTensor)
        z = voxel_min_point_index[:,:,2].type(torch.cuda.LongTensor)
        x[x == -1] = 0
        y[y == -1] = 0
        z[z == -1] = 0

        # Change from grid coordinates to world coordinates
        voxel_min_point = bounding_box_lower + voxel_min_point_index * voxel_size

        # Find the exact intersection points on the isosurface
        intersection_pos = _compute_intersection_pos(grid_subset, intersection_pos_rough,\
                                                    voxel_min_point, x,y,z,\
                                                    ray_direction, voxel_size, mask)
        intersection_pos = intersection_pos * mask.repeat(3,1,1).permute(1,2,0) 

        # Find the round-trip distance from this antenna to each interesction point in the object
        dist_to_intersection = torch.norm(rx_pos_rep - intersection_pos, p=2, dim=2)
        dist_to_intersection = dist_to_intersection * 2
        if is_real_data: dist_to_intersection = dist_to_intersection + 0.15 # TI radar has a 15cm offset

        # Simulate FMCW channels for each reflection and sum them together 
        num_wl = wavelengths.shape[0]
        valid_idx = mask==1
        valid_dists = dist_to_intersection[valid_idx][:, None].repeat(1,num_wl)
        wavelengths_rep = wavelengths[None, :].repeat(valid_dists.shape[0], 1)
        combined_channel += torch.sum(torch.exp(1j*2*torch.pi*valid_dists/wavelengths_rep), axis=0)
    return combined_channel 

def loss_fn(output_ch, target_chs, is_real_data):
    """
    This function computes the cost (or "loss") between a pair of simulated and real channels
    Parameters:
        - output_ch: Simulated channels
        - target_chs: Measured channels
        - is_real_data: Whether data is from real world
    
    Returns loss value
    """
    def fft(x):
        N = len(x)
        k = torch.arange(FFT_SIZE).unsqueeze(0).repeat(FFT_SIZE,1).to(x.device)
        n = torch.arange(FFT_SIZE).unsqueeze(1).repeat(1,FFT_SIZE).to(x.device)
        x_rep = torch.zeros((1,FFT_SIZE,),dtype=torch.complex64).to(device=x.device)
        x_rep[0,:N] = x
        x_rep = x_rep.repeat(FFT_SIZE,1)
        out = torch.sum(x_rep * torch.exp(-1j*2*torch.pi*k*n/FFT_SIZE), dim=1)#/N
        return out

    # Convert to time domain
    output_time_domain = fft(output_ch)
    target_time_domain = fft(target_chs)

    # Remove portions of FFT outside of desired range
    if is_real_data: target_time_domain[MAX_DIST_IDX_REAL:] = 0
    output_time_domain = output_time_domain[MIN_DIST_IDX:MAX_DIST_IDX]
    target_time_domain = target_time_domain[MIN_DIST_IDX:MAX_DIST_IDX]

    # Normalize
    output_time_domain /= torch.norm(output_time_domain)
    target_time_domain /= torch.norm(target_time_domain)

    # Compute loss
    loss = torch.sum(torch.abs(torch.abs(output_time_domain) - torch.abs(target_time_domain))) 
    return loss

def _filter_channels(channels, rx_locs, rx_dirs):
    """ Remove antenna locations with channels that are too weak to provide reliable data """
    # Convert to time domain
    channels_cpu = np.array(channels.to(device='cpu'))
    ch_fft = np.abs(np.fft.fft(channels_cpu, axis=1))
    if is_real_data: ch_fft[:, MAX_DIST_IDX_REAL:] = 0
    ch_fft = ch_fft[:, MIN_DIST_IDX:MAX_DIST_IDX]
    ch_max = np.max(ch_fft, axis=1)
    rx_locs_cpu = rx_locs.cpu()

    # Only keep channels whose peak is above 50% of the peak of the strongest channel
    filter_idx = ch_max > 0.5*np.max(ch_max)

    # Only keep channels whose peak is strong (relative to the average value of the channel). Otherwise, this is likely noise from a very strong but distant reflector
    filter_idx2 =  np.sum(ch_fft, axis=1)/ch_fft.shape[1] < ch_max*0.7 

    # Only keep channels whose peak is not at the bounds of the FFT window. Otherwise, we are likely seeing the side of a stronger peak that lies outside the window
    filter_idx3 =  np.logical_and(np.argmax(ch_fft, axis=1) != 0, np.argmax(ch_fft, axis=1) != ch_fft.shape[1]-1) # Remove ones where max = edge of window

    # Combine the above filters
    filter_idx = np.logical_and(np.logical_and(filter_idx, filter_idx2), filter_idx3)

    # If the above filters were too strict, repeat with weaker requirements
    if np.count_nonzero(filter_idx) < 20:
        filter_idx = ch_max > 0.1*np.max(ch_max)
        filter_idx2 =  np.sum(ch_fft, axis=1)/ch_fft.shape[1] < ch_max*0.7 # 
        filter_idx3 =  np.logical_and(np.argmax(ch_fft, axis=1) != 0, np.argmax(ch_fft, axis=1) != ch_fft.shape[1]-1) # Remove ones where max = edge of window
        filter_idx4 =  np.linalg.norm(rx_locs_cpu[:,:2] - np.tile(np.array([0.15, 0.2])[None,:], (len(rx_locs), 1)), axis=1) < 0.15 # Limit RX locations to center
        filter_idx = np.logical_and(np.logical_and(filter_idx, filter_idx2), filter_idx3)#, filter_idx4)

        # If the above filters were too strict, repeat with weaker requirements
        if np.count_nonzero(filter_idx) < 20:
            filter_idx = ch_max > 0.05*np.max(ch_max)
            filter_idx2 =  np.sum(ch_fft, axis=1)/ch_fft.shape[1] < ch_max*0.7 # 
            filter_idx3 =  np.logical_and(np.argmax(ch_fft, axis=1) != 0, np.argmax(ch_fft, axis=1) != ch_fft.shape[1]-1) # Remove ones where max = edge of window
            filter_idx4 =  np.linalg.norm(rx_locs_cpu[:,:2] - np.tile(np.array([0.15, 0.2])[None,:], (len(rx_locs), 1)), axis=1) < 0.15 # Limit RX locations to center
            filter_idx = np.logical_and(np.logical_and(filter_idx, filter_idx2), filter_idx3)#, filter_idx4)

    # Apply selected filters
    channels = channels[filter_idx, :]
    rx_locs = rx_locs[filter_idx, :]
    rx_dirs = rx_dirs[filter_idx, :]
    return channels, rx_locs, rx_dirs

def _select_tau(all_taus, all_losses):
    """ This function selects the final constant offset. To start, it finds the offset (tau) with minimum loss. Then, it checks if there are other very strong peaks that are higher (i.e., closter to the top) and selects the top peak as the final constant. This ensures that we reconstruct the top surface in cases where we get reflections from both the top and bottom surface. """
    if len(all_losses.shape) == 1: # Case 1: Only 1 RSDF component
        # Start by finding the tau of min loss
        min_idx = np.unravel_index(np.argmin(all_losses), all_losses.shape)
        taus = all_taus[min_idx, :][0]

        # Define a threshold to look for other viable peaks
        threshold = 0.9
        if np.argmin(all_losses) == 0 or np.argmin(all_losses) == len(all_losses)-1: # Check if tau is at the edge of the window. This is typically not desired, so set a lower threshold
            threshold = 0.3

        # Define flipped loss (where higher values are better)
        flipped_loss = np.nanmax(all_losses) - all_losses
        min_loss = flipped_loss[min_idx]

        # Compute threshold of peak search
        threshold_loss = min_loss * threshold

        # Look for other peaks
        peaks, _ = find_peaks(flipped_loss, height=threshold_loss,  prominence=1) 

        # Select the last (i.e., closest to top) peak
        if len(peaks) == 0:
            peaks = [min_idx]
        if min_idx[0] not in peaks and min_idx[0] != 0 and min_idx[0] != len(all_losses)-1: # Account for peak near (but not at edge)
            peaks = np.concatenate((peaks, min_idx))
            peaks.sort()
        taus = all_taus[peaks[-1]]

    else: # Case 2: Multiple RSDF Components
        # Find tau of best loss
        min_idx = np.unravel_index(np.argmin(all_losses), all_losses.shape)

        # Define flipped loss (where higher values are better)
        flipped_loss = np.nanmax(all_losses) - all_losses
        min_loss = flipped_loss[min_idx]

        # Look for other valid peaks in a series of 1D searches (easier than a full N-D search)
        threshold = 0.4
        final_idx = []
        new_idx = list(min_idx)
        for i in range(len(min_idx)): 
            # Create a 1D loss selected at the plane of the min loss
            sel_loss = all_losses
            for j in range(len(min_idx)):
                if j < i: sel_loss = sel_loss[new_idx[j]]
                elif j > i: sel_loss = sel_loss[:, new_idx[j]]
            sel_loss = np.nanmax(sel_loss) - sel_loss

            # Look for other valid peaks
            threshold_loss = max(sel_loss) * threshold
            threshold_loss = max(threshold_loss, 2) 
            height_range = max(sel_loss) - min(sel_loss)
            peaks, _ = find_peaks(sel_loss, height=threshold_loss,  prominence=0.1)
            
            # Update current peak location
            if len(peaks) == 0:
                peaks = [min_idx[i]]
            new_idx[i] = peaks[-1]
        min_idx = new_idx

        # Select final tau
        taus = all_taus
        for i in min_idx:
            taus = taus[i]
    return taus

def _get_final_point_cloud(sdf, taus, coords):
    # Get points for each RSDF Component
    all_points = None
    for sdf_idx in range(sdf.shape[-1]):
        # Select SDF/Tau for this component
        sdf_slice = sdf[:,:,:,sdf_idx]
        tau = taus[sdf_idx]

        # Select the points for this isosurface
        selected_sdf = np.abs(sdf_slice-tau) < 0.001
        selected_points = coords[selected_sdf, :]

        # Combine with previous RSDF points
        if all_points is None:
            all_points = selected_points
        else:
            all_points = np.concatenate((all_points, selected_points), axis=0)
    return all_points

if __name__ == "__main__":
    if not torch.cuda.is_available(): 
        raise Exception('This code requires a GPU to run')

    parser = argparse.ArgumentParser(description='mmWave Image Classification')
    parser.add_argument('--ext', type=str, default='tmp', help='Extension to save/load file with') 
    parser.add_argument('--id', type=str, default='tmp', help='Object ID') 
    parser.add_argument('--name', type=str, default='tmp', help='Object name') 
    parser.add_argument('--is_los', type=str, default="y", help='IS LOS ')
    parser.add_argument("--overwrite_existing", type=str, default="n", help="Overwrite existing file of the same name? If no and a file already exists, it will skip this step.") 
    args = parser.parse_args()
    is_los = (args.is_los=='y' or args.is_los=="Y")
    overwrite_existing = (args.overwrite_existing == 'y' or args.overwrite_existing == 'Y')

    # If applicable, check if a file exists before executing any computation
    if not overwrite_existing:
        try:
            loader = GenericLoader(obj_id, name, is_sim=False, is_los=is_los, exp_num='2') # Note: Only exp 2 is supported at this time
            data = loader.save_optimization_file(radar_type='77_ghz', x_angle=0, y_angle=0, z_angle=0, extra=opt_input_ext)
            if data is not None:
                print(f'The output file already exists and this step was called with the overwrite_existing flag. Skipping this step.')
                sys.exit(0)
        except:
            pass


    # Set up
    torch.backends.cudnn.benchmark = True
    dev = f'cuda:0'
    Tensor = torch.cuda.FloatTensor
    apply_tau_conversion = True
    is_real_data = True


    # Load Data
    sdf, normals, rx_locs, rx_dirs, channels, bounding_box, wavelengths, voxel_size, tau_bounds, loader, num_samples, coords, sum_img = _read_sdf_file(args.id, args.name, is_los, args.ext)
    try:
        voxel_size = Tensor(voxel_size).to(device=dev)
    except:
        pass

    # Move data to GPU
    rx_locs = rx_locs.to(device=dev)
    channels = channels.to(device=dev)
    wavelengths = wavelengths.to(device=dev)
    rx_dirs = rx_dirs.to(device=dev)
    tau_bounds = tau_bounds.to(device=dev)
    normals = normals.to(device=dev)
    sdf = sdf.to(device=dev)
    sdf.requires_grad = False
    normals.requires_grad = False
    rx_locs.requires_grad = False
    channels.requires_grad = False
    wavelengths.requires_grad = False
    rx_dirs.requires_grad = False
    tau_bounds.requires_grad = False

    # Define parameters
    global NUM_SAMPLES, FFT_SIZE, DISTANCES,MAX_DIST_IDX, MAX_DIST_IDX_REAL,MIN_DIST_IDX
    NUM_SAMPLES = num_samples #channels.shape[1]
    FFT_SIZE = 512#NUM_SAMPLES
    wavelengths = wavelengths[:NUM_SAMPLES]
    f_max = 3e8/wavelengths[-1].item()
    max_dist_real = 0.55
    max_dist_all = 0.55
    min_dist = 0.35
    DISTANCES = torch.arange(0, FFT_SIZE) * 3e8 / (2 * (f_max-77.5e9)) * NUM_SAMPLES / FFT_SIZE
    MAX_DIST_IDX = len(DISTANCES[DISTANCES<max_dist_all]) # Don't go beyond 0.5m because will see floor
    MAX_DIST_IDX_REAL = len(DISTANCES[DISTANCES<max_dist_real]) # Don't go beyond 0.5m because will see floor
    MIN_DIST_IDX = len(DISTANCES[DISTANCES<min_dist]) # Don't go beyond 0.5m because will see floor
    width = 1024*2 
    height = 1024*2 
    width_fov = 80
    height_fov = 80 
    num_rx = 50
    bounding_box_lower =  Tensor([bounding_box[0], bounding_box[1], bounding_box[2]])


    # Filter out measurements that are too weak
    # We do this so that when we randomly select a subset of antennas to compute the loss, we do not accidentally select very weak measurements with poor SNR
    channels, rx_locs, rx_dirs = _filter_channels(channels, rx_locs, rx_dirs)


    # Create a grid of all isosurface constants we want to search over
    num_sdf = sdf.shape[-1]
    vectors = []
    step_size = wavelengths[0]/2 if num_sdf < 3 else wavelengths[0]*2 
    for i in range(num_sdf):
        # Create a 1D tensor from min to max with step size of 1
        vector = torch.arange(tau_bounds[i, 0], tau_bounds[i, 1], step_size)
        vectors.append(vector)    
    tau_grid = torch.meshgrid(*vectors, indexing='ij')
    tau_grid = torch.stack(tau_grid, dim=-1) # X x Y x Z x N
    tau_grid = tau_grid[15:,...] #  Remove taus near edge of search grid to improve speed
    num_elements = tau_grid[...,0].numel()  # Number of elements in the tensor
    flat_indices = torch.arange(num_elements)  # Flat indices from 0 to num_elements - 1
    shape = tau_grid.shape[:-1]
    nd_indices = np.unravel_index(flat_indices, shape)
    nd_indices = np.stack(nd_indices, axis=-1) # X x Y x Z x N

    # Empty tensor to hold losses
    all_losses = torch.zeros_like(tau_grid[...,0])

    # Randomly shuffle RX so we can randomly select a subset of them
    rx_shuffle = torch.randperm(len(rx_locs))

    for k in tqdm(range(len(nd_indices))):
        # Create tensor of desired isosurface offsets (taus) and move to GPU
        idxs = nd_indices[k]
        taus = tau_grid
        for i in idxs:
            taus = taus[i]
        taus = taus.to(device=dev)

        # Compute loss over a random subset of antennas
        total_loss = 0
        for l, rx_idx in enumerate(rx_shuffle[:num_rx]): 
            # Estimate channel for isosurface of each RSDF component
            sdf_channels = torch.zeros((num_sdf, NUM_SAMPLES), dtype=torch.complex64).to(device=dev)
            for sdf_idx in torch.arange(num_sdf):
                sdf_channels[sdf_idx, :] = _simulate_channels(bounding_box, bounding_box_lower,\
                            voxel_size, sdf.shape[0], sdf.shape[1], sdf.shape[2], width, height, width_fov, height_fov, sdf, rx_locs[rx_idx], rx_dirs[rx_idx], wavelengths, taus, tau_bounds, is_real_data, 
                            apply_tau_conversion=apply_tau_conversion, sdf_indices=[sdf_idx])

            # Sum channels across all RSDF components
            sum_ch = torch.sum(sdf_channels[:, :], dim=0) 

            # Compute loss and add to total
            loss = loss_fn(sum_ch, channels[rx_idx], is_real_data)
            total_loss += loss

        # Add loss to the correct place in all_losses
        idxs = nd_indices[k]
        if num_sdf > 1:
            arr = all_losses
            for i in idxs[:-1]:
                arr = arr[i]
            arr[idxs[-1]] = total_loss
        else:
            all_losses[nd_indices[k]] = total_loss

    # Select the best constant offset
    taus = _select_tau(tau_grid.detach().cpu().numpy(), all_losses.detach().cpu().numpy())

    # Get the points for these isosurface(s)
    points = _get_final_point_cloud(sdf.detach().cpu().numpy(), taus, coords)

    # We save two files. One is smaller and easier to move, the other contains other auxilary information for debugging
    data = {'taus': taus, 'points': points}
    loader.save_optimization_file(data, radar_type='77_ghz', ext=f'_optimization_result{args.ext}')
    
    data = {'sdf': sdf.detach().cpu().numpy(), 'coords': coords, 'sum_img': sum_img, 'num_samples': num_samples, 
    'all_taus': tau_grid.detach().cpu().numpy(), 
    'all_losses': all_losses.detach().cpu().numpy()}
    loader.save_optimization_file(data, radar_type='77_ghz', ext=f'_full_data{args.ext}')
