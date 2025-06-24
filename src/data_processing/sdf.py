import numpy as np
import scipy.interpolate
import random
import trimesh
from PIL import Image
import time
import scipy
from scipy.constants import c
import json
from scipy import ndimage
from collections import deque as queue
import traceback
import cc3d  
import gc
from scipy.spatial.transform import Rotation as R
import argparse
from skimage import filters
import open3d as o3d
import sys

sys.path.append('..')
from utils import utilities
from utils import generic_loader
from utils.object_information import ObjectInformation, ObjectAttributes, ExperimentAttributes


def _load_sn_file(obj_id, name, is_sim, is_los, radar, ext):
    loader = generic_loader.GenericLoader(obj_id, name, is_sim, is_los=is_los, exp_num='2') # Note: Only exp 2 is supported at this time
    sum_img, (x_locs, y_locs, z_locs), antenna_locs, channels, normals = loader.load_surface_normals(radar_type=radar, ext=ext)
    return sum_img, (x_locs, y_locs, z_locs), antenna_locs, channels, normals, loader


def _separate_binary_image(sum_img, coords_full):
    """
    This function separates the SAR image into a binary image (See Fig. 9b in paper)
    This uses the Li Thresholding algorithm to select the binary image threshold (https://scikit-image.org/docs/0.25.x/auto_examples/developers/plot_threshold_li.html)
    By default, we apply the thresholding to the SAR image power (e.g., |SAR image|^2).
    However, the Li Thresholding may take a significantly long time for some objects, so we terminate it after 1000 iterations, and repeat with a different exponent (e.g., |SAR image|^3) until successful. 
    If it is never successful, we terminate the SDF generation 
    """
    filter_idx = np.zeros_like(sum_img)
    callback_iter = 0

    def _apply_li_threshold(exp):
        """
        Apply Li Thresholding to the image with the given exponential.
        This will stop the Li Threshold if it runs for more than 1000 iterations
        Parameters: exp (float): Exponent to raise the SAR image to
        Returns: success (bool): True if threshold exceeded. False if not
        """
        global callback_iter
        callback_iter = 0
        threshold_fn = filters.threshold_li

        # Define a callback that stops the function after 1000 iterations
        def callback(t):
            global callback_iter
            callback_iter += 1
            if callback_iter > 1000:
                raise Exception('Li Thresholding has exceeded maximum number of iterations. Exiting...')
        kwargs={'iter_callback': callback}

        # Define image as |SAR image|^exp
        abs_img = np.abs(sum_img)**exp

        # Apply modified Li threshold
        nonlocal filter_idx
        try:
            filter_idx = abs_img < threshold_fn(abs_img, **kwargs)
            return True
        except Exception as e: 
            return False

    # Apply Li Thresholding. Start with |SAR|^2. If that fails, continue with higher exponents
    possible_exps = [2, 2.5, 3.5, 4]
    volume_threshold = 0.03
    for exp in possible_exps:
        success = _apply_li_threshold(exp) # Apply Li threshold and check if successful

        # Also check how large the binary image (1) is. If it is very large, consider it a failure
        points = coords_full[np.logical_not(filter_idx)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        volume = pcd.get_minimal_oriented_bounding_box().volume()

        # If this was successful, return the filter idx
        if success and volume <= volume_threshold:
            return filter_idx
    
    # If we have tried all the exponents, ignore the volume component and return if successful
    if success: return filter_idx
    else: sys.exit(0) # This will terminate the SDF generation but allow the other objects called by the bash script to continue

def _find_connected_components(all_normals, x_locs):
    """
    Find 3D connected components in the binary image
    """
    # # Get the binary mask
    mask = ~np.isnan(all_normals[:,:,:,0]) 

    # # Find connected components
    labeled = cc3d.connected_components(mask, connectivity=6)+1

    # # Filter out very small components
    component_min_size = int(0.03/(x_locs[1]-x_locs[0]))
    component_min_size_z = int(0.04/(x_locs[1]-x_locs[0]))
    
    # Find volume of each connected component
    areas = np.array(ndimage.sum(mask, labeled, np.arange(1,labeled.max()+2)))
    
    # Find connected components above a certain size
    valid_labels = np.arange(1,labeled.max()+2)[areas > component_min_size**2*component_min_size_z]
    
    # If no components were found, try again with a smaller threshold
    if len(valid_labels) == 0:
        component_min_size = int(0.015/(x_locs[1]-x_locs[0]))
        component_min_size_z = int(0.02/(x_locs[1]-x_locs[0]))
        areas = np.array(ndimage.sum(mask, labeled, np.arange(1,labeled.max()+2)))
        valid_labels = np.arange(1,labeled.max()+2)[areas > component_min_size**2*component_min_size_z]
    return labeled, valid_labels

def _construct_rsdf_components(all_normals, coords_full, labeled, valid_labels, x_locs):
    """
    This function constructs the RSDF for each connected component
    Our RSDF computation relies on a standard breadth first search
    Note: This process currently assumes square voxels in the image!

    Parameters:
        - all_normals (numpy array): Array of normals
        - coords_full (numpy array): Array of all coordinates of all voxels in the SAR image
        - labeled (numpy array): Defines which voxels belong to which connected component
        - valid_labels (list): List of all valid connected component labels to build an RSDF for

    Returns:
        - RSDF (numpy array): Array with same shape as image. It has RSDF values or nan where there is no valid connected component
    """

    # Prep
    rsdf = np.zeros(all_normals.shape[:-1])
    rsdf[:,:,:] = np.nan
    delta_x = x_locs[1] - x_locs[0] 
    visited = np.zeros_like(labeled)

    # Vectors defining each direction to search in for each iteration of the breadth first search
    dX = [ -1, 0, 1, 0, 0, 0]
    dY = [ 0, 1, 0, -1, 0, 0]
    dZ = [ 0, 0, 0, 0, 1, -1]

    def isValid(visited, normals, labels, x_ind, y_ind, z_ind, current_label):
        """
        Checks whether a given voxel is valid. Specifically, it checks whether:
        - the index of the voxel is within the bounds of the image
        - the voxel has not already been visted in the search
        - the voxel has a valid normal
        - the voxel's component label matches the current label we are searching

        Parameters:
            - visited (numpy array): Boolean array with shape of SAR image telling whether a voxel has been visted or not in the search
            - normals (numpy array): Array of all normals
            - labels (numpy array): Defines which voxels correspond to which connected component labels
            - x_ind: Index of x coordinate of voxel
            - y_ind: Index of y coordinate of voxel
            - z_ind: Index of z coordinate of voxel
            - current_label: Current connected component label we are building the RSDF for

        Returns: 
            - True if the voxel is valid to visit or False if not
        """
        if (x_ind < 0 or y_ind < 0 or z_ind < 0 or x_ind >= visited.shape[0] or y_ind >= visited.shape[1] or z_ind >= visited.shape[2]):
            return False
        if (visited[x_ind,y_ind,z_ind]):
            return False
        if np.isnan(normals[x_ind,y_ind,z_ind,0]):
            return False
        if labels[x_ind,y_ind,z_ind] != current_label:
            return False
        return True

    # Compute the RSDF for each connected component (e.g., each label in valid_labels)
    for component_num in valid_labels: 
        # Find all voxels within this connected component
        idxs = np.argwhere(np.logical_and(labeled==component_num, ~np.isnan(all_normals[:,:,:,0])))

        # Choose an initial pt in this component to define as 0 and start the search
        # We choose a point roughly in the middle of the component
        idxs_mask = np.logical_and(labeled==component_num, ~np.isnan(all_normals[:,:,:,0]))
        coords_sel = coords_full[idxs_mask, :]
        mean_coord = np.mean(coords_sel, axis=0)[None, :] 
        dists = np.linalg.norm((coords_sel - mean_coord), axis=1)
        min_idx = np.argmin(dists) 
        i0, j0, k0 = idxs[min_idx]
        rsdf[i0, j0, k0] = 0

        # Stores indices of the matrix cells
        q = queue()
    
        # Mark the starting cell as visited
        # and push it into the queue
        q.append(( i0, j0, k0 ))
        visited[i0,j0,k0] = True
    
        # Iterate while the queue
        # is not empty
        while (len(q) > 0):
            cell = q.popleft()
            i,j,k = cell
            
            prev = rsdf[i,j,k]
            # Go to the adjacent cells
            for dir_ind in range(len(dX)):
                adjx = i + dX[dir_ind]
                adjy = j + dY[dir_ind]
                adjz = k + dZ[dir_ind]
                vec = np.array([dX[dir_ind],dY[dir_ind], dZ[dir_ind]])
                # Check if the adjacent cell is valid (e.g., within the image bounds, not previously visited, has a valid normal, and within this connected component)
                if (isValid(visited, all_normals, labeled, adjx, adjy, adjz,component_num)):
                    # Add valid voxels to the queue
                    q.append((adjx, adjy, adjz))
                    visited[adjx,adjy, adjz] = True

                    # Compute the RSDF at the adjacent voxel (See Eq. 8 in paper)
                    rsdf[adjx,adjy,adjz] = prev + np.dot(vec*delta_x, all_normals[adjx, adjy, adjz]) 

    return rsdf

def _save_sdf_file(sum_img, x_locs, y_locs, z_locs, antenna_locs, channels, all_normals, loader, rsdf, labeled, valid_labels, coords_full):
    """
    Save RSDF and other all information necessary for isosurface optimization to a file
    """
    params = utilities.get_radar_parameters(radar_type='77_ghz', is_sim=False, aperture_type='almost-square')
    min_f = params["min_f"]; max_f = params["max_f"]; SAMPLES_PER_MEAS = params["num_samples"]
    channels = np.transpose(channels, (0,2,1)) # Nxkx512
    channels = channels.reshape(channels.shape[0]*channels.shape[1], channels.shape[2])
    rx_dirs = np.zeros_like(antenna_locs)
    rx_dirs[:, -1] = 1.0
    data = {}
    data['rsdf'] = rsdf
    data['sum_img'] = sum_img
    data['normals'] = all_normals
    data['rx_locs'] = antenna_locs
    data['rx_dirs'] = rx_dirs
    data['channels'] = channels
    data['component_labels'] = labeled
    data['valid_labels'] = valid_labels
    data['coords'] = coords_full
    data['x_locs'] = x_locs
    data['y_locs'] = y_locs
    data['z_locs'] = z_locs
    data['num_samples'] = SAMPLES_PER_MEAS
    data['bounding_box'] = [x_locs[0], y_locs[0], z_locs[0], x_locs[-1], y_locs[-1], z_locs[-1 ]]
    data['wavelengths'] = c/np.arange(min_f, max_f, (max_f-min_f)/SAMPLES_PER_MEAS)
    data['voxel_size'] = z_locs[1] - z_locs[0] # Assumes square voxels
    loader.save_sdf_file(data, radar_type='77_ghz', ext=ext)

def compute_rsdf(obj_id, name, is_los, ext, overwrite_existing):
    """
    Main function to compute an RSDF for the provided object. This assumes that the normal estimation is already complete
    Parameters:
        obj_id (str): ID of object
        name (str): Name of object
        is_los (bool): Use LOS experiment? False will use NLOS Experiment
        ext (str): Ext with which to load the normal file and save the SDF file
        overwrite_existing (bool): Whether to overwrite existing files, or skip the computation if the file already exists
    """
    # If applicable, check if a file already exists before running any computation
    if not overwrite_existing:
        try:
            loader = generic_loader.GenericLoader(obj_id, name, is_sim=False, is_los=is_los, exp_num='2')
            data = loader.load_sdf_file(radar_type='77_ghz', ext=ext)
            if data is not None:
                print(f'The output file already exists and this step was called with the overwrite_existing flag. Skipping this step.')
                return
        except:
            pass

    # # # Step 0: Load sum image and prep data
    sum_img, (x_locs, y_locs, z_locs), antenna_locs, channels, all_normals, loader  = _load_sn_file(obj_id, name, False, is_los, '77_ghz', ext)
    mesh1, mesh2, mesh3 = np.meshgrid(y_locs, x_locs, z_locs)
    coords_full = np.concatenate((mesh2[...,np.newaxis], mesh1[...,np.newaxis],mesh3[...,np.newaxis]), axis=-1) # Define full numpy array of coordinates of every voxel

    # # # Step 1: Separate image into a binary image (See Fig. 9b in the paper)
    filter_idx = _separate_binary_image(sum_img, coords_full)
    all_normals[filter_idx] = np.nan # Remove normals that are 0 in the binary image
    all_normals /= np.linalg.norm(all_normals, axis=-1)[...,np.newaxis].repeat(3, axis=-1) # Normalize normals to unit vectors if they weren't already

    # # # Step 2: Find connected components in the binary image (See Fig. 9c in the paper)
    labeled, valid_labels = _find_connected_components(all_normals, x_locs)

    # # # Step 3: Build RSDFs for each connected component (See Fig. 9d in the paper)
    rsdf = _construct_rsdf_components(all_normals, coords_full, labeled, valid_labels, x_locs)

    # # # Step 4: Save RSDF file
    _save_sdf_file(sum_img, x_locs, y_locs, z_locs, antenna_locs, channels, all_normals, loader, rsdf, labeled, valid_labels, coords_full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that  run 24ghz robot imager")
    parser.add_argument("--name", type=str, default="hammer" , help="Object Name")
    parser.add_argument("--id", type=str, default="048", help="Object ID")
    parser.add_argument("--ext", type=str, default="", help="Extension")
    parser.add_argument("--is_los", type=str, default="y", help="Plot GT?")
    parser.add_argument("--overwrite_existing", type=str, default="n", help="Overwrite existing file of the same name? If no and a file already exists, it will skip this step.")
    args = parser.parse_args()

    def convert_str_to_bool(arg_input):
        return True if (arg_input == 'y' or arg_input== 'Y') else False
    obj_id = args.id
    obj_name = args.name
    ext = args.ext
    overwrite_existing = convert_str_to_bool(args.overwrite_existing)
    is_los = convert_str_to_bool(args.is_los)

    # Fill in missing info
    obj_info = ObjectInformation()
    assert not (obj_name == '' and obj_id == ''), "Both name and ID can't be empty"
    obj_id, obj_name = obj_info.fill_in_identifier_sep(obj_name, obj_id)

    # Compute RSDF and save result to a file
    compute_rsdf(obj_id, obj_name, is_los, ext, overwrite_existing)
    
