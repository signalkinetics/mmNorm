import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import threading
import gc
import time
from scipy.fft import fft, ifft
import math

threadLock = threading.Lock()
class GPUThread(threading.Thread):
    """
    This class starts computation on the GPU. It manages the computation on a separate CPU thread to allow for multiple GPUs simultaneously. 
    It is based on the example code here:  https://shephexd.github.io/development/2017/02/19/pycuda.html
    """
    def __init__(self, number, sum_image_r, sum_image_i, valid_idx_x, valid_idx_y, valid_idx_z, x_locs, y_locs, z_locs, antenna_locs_flat, meas_real, meas_imag, rx_offset, slope, wavelength, fft_spacing, num_x, num_y, num_z, num_ant, num_rx_ant, num_valid_idx,samples_per_meas,image_size,threads_per_block, grid_dim, num_gpus, is_ti_radar):
        threading.Thread.__init__(self)
        self.p_xyz_r = sum_image_r
        self.p_xyz_i = sum_image_i
        self.surface_normals_x = np.empty((num_valid_idx,), dtype=np.float32)
        self.surface_normals_x[:] = 0
        self.surface_normals_y = np.empty((num_valid_idx,), dtype=np.float32)
        self.surface_normals_y[:] = 0
        self.surface_normals_z = np.empty((num_valid_idx,), dtype=np.float32)
        self.surface_normals_z[:] = 0
        self.number = number
        self.x_locs = x_locs
        self.y_locs = y_locs 
        self.z_locs = z_locs  
        self.antenna_locs_flat = antenna_locs_flat  
        self.meas_real = meas_real  
        self.meas_imag = meas_imag  
        self.rx_offset = rx_offset  
        self.slope = slope  
        self.wavelength = wavelength 
        self.fft_spacing = fft_spacing
        self.num_x = num_x 
        self.num_y = num_y  
        self.num_z = num_z  
        self.num_ant = num_ant  
        self.num_rx_ant = num_rx_ant
        self.threads_per_block = threads_per_block  
        self.grid_dim = grid_dim 
        self.is_ti_radar=is_ti_radar
        self.image_size=image_size
        self.valid_idx_x = valid_idx_x
        self.valid_idx_y = valid_idx_y
        self.valid_idx_z = valid_idx_z
        self.num_valid_idx=num_valid_idx
        self.samples_per_meas = samples_per_meas

    def run(self):
        """
        Runs the computation on the GPU
        """
        # Set up GPU context
        self.dev = drv.Device(self.number)
        self.ctx = self.dev.make_context()

        mod = None
        try:
            mod = SourceModule("""
            // This file computes surface estimates on the GPU. This contains the bulk of the mmWave surface normal estimation process (e.g., Eq. 6 in the mmNorm paper)
#include <cuda_runtime.h>



#define SPEED_LIGHT 2.99792458e8
const double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

#define mult_r(a,b,c,d)(a*c-b*d)
#define mult_i(a,b,c,d)(a*d+b*c)
  

extern "C"
__global__ void cuda_sn_est(float* device_normals_x, float* device_normals_y, float* device_normals_z, 
                            int* valid_idx_x, int* valid_idx_y, int* valid_idx_z,
                           float* device_p_xyz_r, 
                           float* device_p_xyz_i, 
                           float* device_x_locs,
                           float* device_y_locs,
                           float* device_z_locs,
                           float* device_antenna_locs, 
                           float* device_measurements_r,
                           float* device_measurements_i, 
                           float* rx_offsets, float slope, float wavelength, float fft_spacing, 
                           int NUM_X, int NUM_Y, int NUM_Z, int NUM_ANTENNAS, int NUM_RX_ANTENNAS, int NUM_VALID_IDX, int SAMPLES_PER_MEAS, int start_ind, int is_ti_radar ) {
    /*
    This kernel function mmWave surface normal estimates on the GPU. It is called across different voxels in the image to compute normal estimates for the full image.
    We do not compute normal estimates for every voxel, since many voxels contain very low values in the SAR image and thus have too poor SNR for surface normal estimation (since many of these voxels do not have an object's surface anyway).
    Instead, we define a list of valid voxels, and only compute normal estimates for these voxels
    Parameters:
        These parameters are outputs:
            - device_normals_x: Stores the x coordinate of the normal estimates
            - device_normals_y: Stores the y coordinate of the normal estimates
            - device_normals_z: Stores the z coordinate of the normal estimates
    
        The remaining parameters are all inputs:
            - valid_idx_x: Array mapping an index to one of the valid voxels. This array returns the index of the x coordiate of that valid voxel.
            - valid_idx_y: Array mapping an index to one of the valid voxels. This array returns the index of the y coordiate of that valid voxel.
            - valid_idx_z: Array mapping an index to one of the valid voxels. This array returns the index of the z coordiate of that valid voxel.
            - device_p_xyz_r: The real value of the SAR image
            - device_p_xyz_i: The imaginary value of the SAR image
            - device_x_locs: the X locations of all voxels in the SAR image
            - device_y_locs: the Y locations of all voxels in the SAR image
            - device_z_locs: the Z locations of all voxels in the SAR image
            - device_antenna_locs: the locations of the TX antenna for each measurement. If using multiple RX antennas for each TX, this should include repeated TX location for each RX antenna (see py_image_gpu for more details). 
            - device_measurements_r: The real value of the radar measurements. This should be after an FFT operation.
            - device_measurements_i: The imaginary value of the radar measurements. This should be after an FFT operation.
            - rx_offsets: The 3D offsets between the TX antenna and the RX antennas 
            - slope: The slope of the radars chirp
            - wavelength: The starting wavelength of the radars chirp
            - fft_spacing: The spacing (in m) between two points in the FFT of the radar data
            - NUM_X: the number of x voxels 
            - NUM_Y: the number of y voxels
            - NUM_Z: the number of z voxels
            - NUM_ANTENNAS: the number of antenna locations
            - NUM_RX_ANTENNAS: the number of RX antennas in use
            - NUM_VALID_IDX: the number of valid voxels
            - SAMPLES_PER_MEAS: the number of samples in each radar measurement
            - start_ind: offset to apply to the block index. Used when splitting computation across multiple GPUs
            - is_ti_radar: whether this data comes from a TI radar (to apply correction)
    */

    // Find the current index from the GPU block/thread index
    int ind = (blockIdx.x * blockDim.x + threadIdx.x) + start_ind;
    if (ind >= NUM_VALID_IDX || ind < 0) { return; }

    // Convert the index to the location of a valid voxel
    int x = valid_idx_x[ind];
    int y = valid_idx_y[ind];
    int z = valid_idx_z[ind];
    float x_loc = device_x_locs[x];
    float y_loc = device_y_locs[y];
    float z_loc = device_z_locs[z];

    // Read the SAR value of this voxel
    float image_r = device_p_xyz_r[x*NUM_Y*NUM_Z+y*NUM_Z+z];
    float image_i = device_p_xyz_i[x*NUM_Y*NUM_Z+y*NUM_Z+z];
    float image_mag = sqrtf(image_r * image_r + image_i * image_i);

    // Compute surface normal for this voxel. Need to compute weighted sum across all antennas
    float sn_x = 0;
    float sn_y = 0;
    float sn_z = 0;
    for (uint i = 0; i < NUM_ANTENNAS; i++) {
        // Find location of TX/RX antenna
        float x_antenna_loc = device_antenna_locs[i*3+0];
        float y_antenna_loc = device_antenna_locs[i*3+1];
        float z_antenna_loc = device_antenna_locs[i*3+2];
        float antenna_x_diff = x_loc - x_antenna_loc;
        float antenna_y_diff = y_loc - y_antenna_loc;
        float antenna_z_diff = z_loc - z_antenna_loc;
        int rx_num = i%NUM_RX_ANTENNAS;
        float rx_offset_x = rx_offsets[rx_num*3+0];
        float rx_offset_y = rx_offsets[rx_num*3+1];
        float rx_offset_z = rx_offsets[rx_num*3+2];

        // ---------- Compute SAR image component using this single antenna (Eq. 4 in paper) -------------------
        // Find distance from TX -> voxel -> RX
        float forward_dist = sqrtf(antenna_x_diff * antenna_x_diff + 
                                        antenna_y_diff * antenna_y_diff + 
                                        antenna_z_diff * antenna_z_diff);
        float back_dist = sqrtf((antenna_x_diff - rx_offset_x)* (antenna_x_diff - rx_offset_x) + 
                                    (antenna_y_diff - rx_offset_y) * (antenna_y_diff - rx_offset_y) + 
                                    (antenna_z_diff - rx_offset_z) * (antenna_z_diff - rx_offset_z));
        float distance = forward_dist + back_dist;
        if (is_ti_radar != 0){ 
            distance += 0.15; // The TI radars have an offset of 15cm that needs to be accounted for
        }

        // Check if distance is valid
        if (distance < 0 || distance > fft_spacing*SAMPLES_PER_MEAS) {
            continue;
        }

        // Find which bin in FFT corresponds to this distance
        int dist_bin = floorf(distance / fft_spacing/2);

        // Select the appropriate measurement, and coorelate with the AoA phase
        float real_meas = device_measurements_r[i*SAMPLES_PER_MEAS+dist_bin];
        float imag_meas = device_measurements_i[i*SAMPLES_PER_MEAS+dist_bin];
        float real_phase = std::cos(-2 * pi * distance / wavelength);
        float imag_phase = std::sin(-2 * pi * distance / wavelength);
        float sum_r = mult_r(real_meas, imag_meas, real_phase, imag_phase);
        float sum_i = mult_i(real_meas, imag_meas, real_phase, imag_phase);

        // ---------- Compute weighted candidate normal vector and add to current sum (Eq. 3/6 in paper) -------------------
        // Virtual antenna between TX/RX
        float virtual_antenna_x =  x_antenna_loc + rx_offset_x / 2; 
        float virtual_antenna_y =  y_antenna_loc + rx_offset_y / 2; 
        float virtual_antenna_z =  z_antenna_loc + rx_offset_z / 2; 

        // Vector from antenna to voxel location
        float vec_x = virtual_antenna_x - x_loc;
        float vec_y = virtual_antenna_y - y_loc;
        float vec_z = virtual_antenna_z - z_loc;

        // Compute vote (Eq. 6)
        float weight = (sum_r * image_r) + (sum_i * image_i);
        weight /= image_mag;

        // Add weighted candidate vector to current weighted sum
        sn_x += (vec_x*weight);
        sn_y += (vec_y*weight);
        sn_z += (vec_z*weight);
    }
    // Save normal estimate
    device_normals_x[ind] = sn_x;
    device_normals_y[ind] = sn_y;
    device_normals_z[ind] = sn_z;
}

int main() {
    // cuda_hello<<<1,1>>>(); 
    return 0;
}
    """)
        except:
            filepath = f"{utilities.get_root_path()}/src/data_processing/cuda/surface_normal_est.o"
            mod = drv.module_from_file(filepath) 

        # Call CUDA function
        self.cuda_image = mod.get_function("cuda_sn_est")
        start_ind = int(self.grid_dim) * int(self.threads_per_block) * self.number
        print(f"Starting {self.number} with start ind {start_ind}")
        # Everything needs to be float32 for cuda
        self.cuda_image(
                drv.Out(self.surface_normals_x), 
                drv.Out(self.surface_normals_y),
                drv.Out(self.surface_normals_z),
                drv.In(self.valid_idx_x.astype(np.int32)), 
                drv.In(self.valid_idx_y.astype(np.int32)), 
                drv.In(self.valid_idx_z.astype(np.int32)), 
                drv.In(self.p_xyz_r.astype(np.float32)), 
                drv.In(self.p_xyz_i.astype(np.float32)), 
                drv.In(self.x_locs.astype(np.float32)), 
                drv.In(self.y_locs.astype(np.float32)), 
                drv.In(self.z_locs.astype(np.float32)), 
                drv.In(self.antenna_locs_flat.astype(np.float32)), 
                drv.In(self.meas_real.astype(np.float32)), 
                drv.In(self.meas_imag.astype(np.float32)), 
                drv.In(self.rx_offset.astype(np.float32)), np.float32(self.slope), np.float32(self.wavelength), np.float32(self.fft_spacing),
                np.int32(self.num_x), np.int32(self.num_y), np.int32(self.num_z), np.int32(self.num_ant),np.int32(self.num_rx_ant),np.int32(self.num_valid_idx),np.int32(self.samples_per_meas),np.int32(start_ind),np.int32(self.is_ti_radar),
                block=(int(self.threads_per_block),1,1), grid=(int(self.grid_dim),1,1))

        # Clean up GPU context
        print(f"successful exit from thread {self.number}")
        self.ctx.pop()
        self.ctx.detach()
        del self.ctx
        self.ctx = None
        gc.collect()
        
    
    def get_res(self):
        """
        Returns the resulting SAR image
        Returns:
            self.surface_normals_x: x coordinate of surface normal estimates
            self.surface_normals_y: y coordinate of surface normal estimates
            self.surface_normals_z: z coordinate of surface normal estimates
        """
        return self.surface_normals_x, self.surface_normals_y, self.surface_normals_z

def _prep_inputs(radar_data, rx_offset, antenna_locs, use_4_rx):
    """
    Reformats inputs to prepare for input to GPU
    Parameters:
        - radar_data (numpy array): Radar measurement data starting as a (#Meas, #Sample/Meas, #RXAntenna) shaped array
        - rx_offset (list): 2D list of offsets from TX antenna to each RX antenna. Shape should be (#RXAntenna, 3)
        - antenna_locs: TX antenna locations of each measurement. Should be one location per measurement
        - use_4_rx: If the data contains 4 RX antennas 
    """
    if not use_4_rx: 
        rx_offset = np.array([rx_offset]) # Expecting array now
        radar_data = np.array(radar_data).flatten()
    else:
        # Reorder radar data to be (#Meas, #RXAntenna, #Sample/Meas)
        radar_data = np.transpose(radar_data, (0,2,1)) # Nxkx512

        rx_offset = np.array(rx_offset).flatten() 

        # Repeat TX locations for each RX location, such that there is 1 TX location per (#Meas x #RXAntenna)
        antenna_locs = np.repeat(antenna_locs, radar_data.shape[1], axis=0)

    return rx_offset, radar_data, antenna_locs


def _sn_est_cuda(sum_image, x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offset, slope, wavelength, bandwidth, num_samples, use_4_rx=False, is_ti_radar=False, initial_filter_percent=0.05):
    '''
    Estimates surface normals on GPU
    Parameters: 
        - sum_image (numpy array): SAR image
        - x_locs (numpy array): X locations to process SAR image at
        - y_locs (numpy array): Y locations to process SAR image at
        - z_locs (numpy array): Z locations to process SAR image at
        - antenna_locs (numpy array): Locations of TX antennas at each measurement
        - radar_data (numpy array): Radar measurement data shaped as (#Meas, #Sample/Meas, #RXAntenna) or (#Meas, #Sample/Meas)
        - rx_offset (numpy array): Offset from TX to each of the RX antennas. Shape should be (#RXAntenna, 3)
        - slope (float): Slope of the radar chirp
        - wavelength (float): Starting wavelength of the radar chirp
        - bandwidth (float): Bandwidth of the radar chirp
        - num_samples (float): Number of samples in the original radar chirp (note: not the number of samples after the FFT)
        - use_4_rx (bool): If the data contains 4 RX antennas
        - is_ti_radar (bool): If the data is real data coming from a TI radar (e.g., the 77GHz radar)
        - use_interpolated_processing (bool): Whether or not to use interpolated processing (e.g., backprojection) instead of a complete matched filter. Backprojection will be significantly faster in most scenarios
    '''
    # Find number of available GPUs
    num_gpus = drv.Device.count()
    print(f'Detected {num_gpus} CUDA Capable device(s)')

    # Prepare input to CUDA
    norm_factor = antenna_locs.shape[0]
    samples_per_meas = radar_data.shape[1]
    rx_offset_flat, measurements, antenna_locs = _prep_inputs(radar_data, rx_offset, antenna_locs, use_4_rx)
    meas_real = np.ascontiguousarray(measurements.real)
    meas_imag = np.ascontiguousarray(measurements.imag)
    antenna_locs_flat = np.array(antenna_locs).flatten()
    sum_image_r = np.real(sum_image).flatten()
    sum_image_i = np.imag(sum_image).flatten()

    # Select which voxels to estimate surface normals for.
    # Only select those with a SAR image value above some percentage of the maximum SAR image value
    filter_idx = np.abs(sum_image)>(np.max(np.abs(sum_image))*initial_filter_percent)
    mesh1, mesh2, mesh3 = np.meshgrid(np.arange(len(y_locs)), np.arange(len(x_locs)), np.arange(len(z_locs)))
    coords_full = np.concatenate((mesh2[...,np.newaxis], mesh1[...,np.newaxis],mesh3[...,np.newaxis]), axis=-1)
    coords_full = coords_full[filter_idx]
    valid_idx_x = coords_full[:,0]
    valid_idx_y = coords_full[:,1]
    valid_idx_z = coords_full[:,2]
    num_valid_idx = valid_idx_y.shape[0]
    
    # Prepare input to CUDA
    num_x = len(x_locs)
    num_y = len(y_locs)
    num_z = len(z_locs)
    num_ant = len(antenna_locs)
    num_rx_ant = len(rx_offset)
    threads_per_block = 512
    image_size = num_valid_idx#num_x*num_y*num_z
    grid_dim = int(image_size/ threads_per_block / num_gpus) + 1
    fft_spacing = np.float32(3e8/(2*bandwidth)*num_samples/(radar_data.shape[1]))
    print(f'Starting GPU computation')

    # Start a new thread for each GPU. Each thread is responsible for starting, stopping, and cleaning the CUDA code 
    gpu_thread_list = []
    for i in range(num_gpus):
        gpu_thread = GPUThread(i, sum_image_r, sum_image_i, valid_idx_x, valid_idx_y, valid_idx_z, x_locs, y_locs, z_locs, antenna_locs_flat, meas_real, meas_imag, rx_offset_flat, slope, wavelength, fft_spacing, num_x, num_y, num_z, num_ant, num_rx_ant, num_valid_idx, samples_per_meas, image_size, threads_per_block, grid_dim, num_gpus,is_ti_radar)
        gpu_thread.start()
        gpu_thread_list.append(gpu_thread)

    # Get outputs from each GPU when they are done
    normals_x = []
    normals_y = []
    normals_z = []
    for thread in gpu_thread_list:
        thread.join()
        res_x, res_y, res_z = thread.get_res()
        normals_x.append(res_x)
        normals_y.append(res_y)
        normals_z.append(res_z)

    # Sum outputs across all GPUs
    normals_x = np.sum(normals_x, axis=0) 
    normals_y = np.sum(normals_y, axis=0) 
    normals_z = np.sum(normals_z, axis=0) 

    # Combine X/Y/Z coordinates into 1 numpy array
    normals = np.concatenate((normals_x[:,np.newaxis], normals_y[:,np.newaxis], normals_z[:,np.newaxis]), axis=-1)
    
    # Reformat normals to be in same shape as original SAR image for simplicity. Voxels that were not computed are set to NaN 
    all_normals = np.zeros((num_x, num_y, num_z, 3))
    all_normals[:] = np.nan
    all_normals[filter_idx] = normals

    del gpu_thread_list
    return all_normals


def normal_estimates_cuda(sum_image, x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offset, slope, wavelength, bandwidth, num_samples, use_4_rx=False, is_ti_radar=False, initial_filter_percent=0.05, use_interpolated_processing=True):
    '''
    Computes mmWave surface normal estimates on GPU
    If needed, this function will break the processing into multiple separate calls to the GPU to limit the GPU memory of each call. 
    It will also parallelize across all available GPUs.
    Parameters: 
        - sum_image (numpy array): mmWave SAR image computed at same voxels as desired normal estimates. Note: this should be computed with the same processing type as the normal estimation (i.e., both interpolated processing)
        - x_locs (numpy array): X locations to process SAR image at
        - y_locs (numpy array): Y locations to process SAR image at
        - z_locs (numpy array): Z locations to process SAR image at
        - antenna_locs (numpy array): Locations of TX antennas at each measurement
        - radar_data (numpy array): Radar measurement data shaped as (#Meas, #Sample/Meas, #RXAntenna) or (#Meas, #Sample/Meas)
        - rx_offset (numpy array): Offset from TX to each of the RX antennas. Shape should be (#RXAntenna, 3)
        - slope (float): Slope of the radar chirp
        - wavelength (float): Starting wavelength of the radar chirp
        - bandwidth (float): Bandwidth of the radar chirp
        - use_4_rx (bool): If the data contains 4 RX antennas
        - is_ti_radar (bool): If the data is real data coming from a TI radar (e.g., the 77GHz radar)
        - return_phase_centers (bool): Whether to compute and return the virtual phase centers (between TX and RX) of each measurement location
        - initial_filter_percent (bool): Only compute normal estimates for voxels with SAR values above this percentage of maximum value of the image. Since we dont need to compute normal estimates of voxels with very low SNRs (e.g., where there is no object), this significantly reduces computation time
        - use_interpolated_processing (bool): Whether or not to use interpolated processing (e.g., backprojection) instead of a complete matched filter. Backprojection will be significantly faster in most scenarios
    '''
    if not use_interpolated_processing:
       raise Excpetion("Surface normal estimation is not currently implemented without interpolated processing!")
    else:
        # If using the interpolated processing, the memory consumption will be higher. Therefore, break the call into multiple different groups as needed
        max_num_ant = 4096*32 # must be multiple of max_rx, reduce this to reduce amount of GPU memory used by each group
        num_rx = len(rx_offset)
        num_ant_groups = math.ceil(antenna_locs.shape[0]*num_rx/max_num_ant)

        # Divide each radar location into different groups, compute the normal estimates for each group and combine 
        for i in range(num_ant_groups):
            loc_subset = antenna_locs[i*max_num_ant//num_rx:(i+1)*max_num_ant//num_rx]
            meas_subset = radar_data[i*max_num_ant//num_rx:(i+1)*max_num_ant//num_rx]
            normals_subset = _sn_est_cuda(sum_image, x_locs, y_locs, z_locs, loc_subset, meas_subset, rx_offset, slope, wavelength, bandwidth, num_samples, use_4_rx, is_ti_radar, initial_filter_percent)
            
            # Combine normal estimates across groups
            if i == 0:
                normals = normals_subset
            else:
                normals += normals_subset
        
    return normals

