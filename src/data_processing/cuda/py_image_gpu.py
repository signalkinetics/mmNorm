import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import threading
import gc
import sys
import math
import copy

sys.path.append('..')
from utils import utilities

threadLock = threading.Lock()
class GPUThread(threading.Thread):
    """
    This class starts computation on the GPU. It manages the computation on a separate CPU thread to allow for multiple GPUs simultaneously. 
    It is based on the example code here:  https://shephexd.github.io/development/2017/02/19/pycuda.html
    """
    def __init__(self, number, x_locs, y_locs, z_locs, antenna_locs_flat, meas_real, meas_imag, rx_offset, slope, wavelength, fft_spacing, num_x, num_y, num_z, num_ant, num_rx_ant, samples_per_meas, is_ti_radar, use_interpolated_processing, threads_per_block, grid_dim, num_gpus):
        threading.Thread.__init__(self)
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
        self.num_x = num_x 
        self.num_y = num_y  
        self.num_z = num_z  
        self.num_ant = num_ant  
        self.num_rx_ant = num_rx_ant
        self.threads_per_block = threads_per_block  
        self.grid_dim = grid_dim 
        self.p_xyz_r = np.empty((num_x*num_y*num_z,), dtype=np.float32)
        self.p_xyz_i = np.empty((num_x*num_y*num_z,), dtype=np.float32)
        self.p_xyz_r[:] = 0
        self.p_xyz_i[:] = 0
        self.samples_per_meas=samples_per_meas
        self.is_ti_radar=is_ti_radar
        self.use_interpolated_processing=use_interpolated_processing
        self.fft_spacing=fft_spacing

    def run(self):
        """
        Runs the computation on the GPU
        """
        # Set up GPU context
        self.dev = drv.Device(self.number)
        self.ctx = self.dev.make_context()
        try:
            mod = SourceModule("""
#include <cuda_runtime.h>

#define SPEED_LIGHT 2.99792458e8
const double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

// For multiplying two complex numbers. Two functions for the real and imaginary results 
#define mult_r(a,b,c,d)(a*c-b*d)
#define mult_i(a,b,c,d)(a*d+b*c)
  

extern "C"
__global__ void cuda_image(float* device_p_xyz_r, 
                           float* device_p_xyz_i, 
                           float* device_x_locs,
                           float* device_y_locs,
                           float* device_z_locs,
                           float* device_antenna_locs, 
                           float* device_measurements_r,
                           float* device_measurements_i, 
                           float* rx_offsets, float slope, float wavelength,  float fft_spacing, 
                           int NUM_X, int NUM_Y, int NUM_Z, int NUM_ANTENNAS, int NUM_RX_ANTENNAS, int SAMPLES_PER_MEAS, int start_ind, int is_ti_radar, int use_interpolated_processing ) {
    /*
    This kernel function computes the SAR image value for a single voxel. It is called across all voxels to compute the full location. 
    Parameters:
        - device_p_xyz_r: Stores the real value of the SAR image
        - device_p_xyz_i: Stores the imaginary value of the SAR image
        - device_x_locs: the X locations of all voxels to compute the SAR image for
        - device_y_locs: the Y locations of all voxels to compute the SAR image for
        - device_z_locs: the Z locations of all voxels to compute the SAR image for
        - device_antenna_locs: the locations of the TX antenna for each measurement. If using multiple RX antennas for each TX, this should include repeated TX location for each RX antenna (see py_image_gpu for more details). 
        - device_measurements_r: The real value of the radar measurements
        - device_measurements_i: The imaginary value of the radar measurements
        - rx_offsets: The 3D offsets between the TX antenna and the RX antennas 
        - slope: The slope of the radars chirp
        - wavelength: The starting wavelength of the radars chirp
        - NUM_X: the number of x voxels 
        - NUM_Y: the number of y voxels
        - NUM_Z: the number of z voxels
        - NUM_ANTENNAS: the number of antenna locations
        - NUM_RX_ANTENNAS: the number of RX antennas in use
        - SAMPLES_PER_MEAS: the number of samples in each radar measurement
        - start_ind: offset to apply to the block index. Used when splitting computation across multiple GPUs
        - is_ti_radar: whether this data comes from a TI radar (to apply correction)
        - use_interpolated_processing: If this is not equal to 0, we will apply the interpolated (faster, approximate) processing. Otherwise, we will apply the slower, more accurate processing
    */

    // Split block index into x/y/z index and check it is within the image bounds
    int ind = (blockIdx.x * blockDim.x + threadIdx.x) + start_ind;
    int z = ind % NUM_Z;
    ind /= NUM_Z;
    int y = ind % NUM_Y;
    ind /= NUM_Y;
    int x = ind % NUM_X;
    if ((blockIdx.x * blockDim.x + threadIdx.x)+start_ind >= NUM_X*NUM_Y*NUM_Z || x < 0 || y < 0 || z < 0 || x >= NUM_X || y >= NUM_Y || z >= NUM_Z) { return; }
    
    // Convert index to an x/y/z coordinate of a voxel
    float x_loc = device_x_locs[x];
    float y_loc = device_y_locs[y];
    float z_loc = device_z_locs[z];


    // Compute the image value for this voxel. Need to sum the result across all antennas 
    float sum_r = 0;
    float sum_i = 0;
    for (uint i = 0; i < NUM_ANTENNAS; i++) {
        // Find the distance from TX antenna -> voxel -> RX antenna
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
        float forward_dist = sqrtf(antenna_x_diff * antenna_x_diff + 
                                        antenna_y_diff * antenna_y_diff + 
                                        antenna_z_diff * antenna_z_diff);
        float back_dist = sqrtf((antenna_x_diff - rx_offset_x)* (antenna_x_diff - rx_offset_x) + 
                                    (antenna_y_diff - rx_offset_y) * (antenna_y_diff - rx_offset_y) + 
                                    (antenna_z_diff - rx_offset_z) * (antenna_z_diff - rx_offset_z));
        float distance = forward_dist + back_dist;

        // The TI radars have an offset of 15cm that needs to be accounted for
        if (is_ti_radar != 0){ 
            distance += 0.15;
        }

        if (use_interpolated_processing != 0){ // Apply faster but approximate coorelation
            // Check that our distance is valid
            if (distance < 0 || distance > fft_spacing*SAMPLES_PER_MEAS) {
                continue;
            }

            // Find which bin within the range FFT this distance falls
            int dist_bin = floorf(distance / fft_spacing/2);

            // Select the appropriate measurement, and coorelate with the AoA phase
            float real_meas = device_measurements_r[i*SAMPLES_PER_MEAS+dist_bin];
            float imag_meas = device_measurements_i[i*SAMPLES_PER_MEAS+dist_bin];
            float real_phase = std::cos(-2 * pi * distance / wavelength);
            float imag_phase = std::sin(-2 * pi * distance / wavelength);
            sum_r += mult_r(real_meas, imag_meas, real_phase, imag_phase);
            sum_i += mult_i(real_meas, imag_meas, real_phase, imag_phase);
        } else { // Apply more accurate but slower correlation
            // Compute the sum across all samples in this measurement
            for (uint j = 0; j < SAMPLES_PER_MEAS; j++) {
                // Correlate this sample with the complex value of the expected measurement (see references for more detail)
                float real_meas = device_measurements_r[i*SAMPLES_PER_MEAS+j];
                float imag_meas = device_measurements_i[i*SAMPLES_PER_MEAS+j];
                float real_phase = std::cos(-2 * pi * distance / wavelength + -2 * pi * distance * j * slope / SPEED_LIGHT);
                float imag_phase = std::sin(-2 * pi * distance / wavelength + -2 * pi * distance * j * slope / SPEED_LIGHT);
                sum_r += mult_r(real_meas, imag_meas, real_phase, imag_phase);
                sum_i += mult_i(real_meas, imag_meas, real_phase, imag_phase);
            }
        }
    
    }
    // Save resulting voxel value (real/imaginary values)
    device_p_xyz_r[x*NUM_Y*NUM_Z+y*NUM_Z+z] = sum_r;
    device_p_xyz_i[x*NUM_Y*NUM_Z+y*NUM_Z+z] = sum_i;
}""")
        except:
            filepath = f"{utilities.get_root_path()}/src/data_processing/cuda/imaging_gpu.o"
            mod = drv.module_from_file(filepath) 
        self.cuda_image = mod.get_function("cuda_image")

        # Call CUDA function
        print(f"Starting GPU {self.number}")
        start_ind = int(self.grid_dim) * int(self.threads_per_block) * self.number # When using multiple GPUs, offset their computations
        # Everything needs to be float32 for cuda
        self.cuda_image(drv.Out(self.p_xyz_r), 
                drv.Out(self.p_xyz_i), 
                drv.In(self.x_locs.astype(np.float32)), 
                drv.In(self.y_locs.astype(np.float32)), 
                drv.In(self.z_locs.astype(np.float32)), 
                drv.In(self.antenna_locs_flat.astype(np.float32)), 
                drv.In(self.meas_real.astype(np.float32)), 
                drv.In(self.meas_imag.astype(np.float32)), 
                drv.In(self.rx_offset.astype(np.float32)), np.float32(self.slope), np.float32(self.wavelength), np.float32(self.fft_spacing),
                np.int32(self.num_x), np.int32(self.num_y), np.int32(self.num_z), np.int32(self.num_ant),np.int32(self.num_rx_ant),np.int32(self.samples_per_meas),np.int32(start_ind),np.int32(self.is_ti_radar),np.int32(self.use_interpolated_processing),
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
            self.p_xyz_r: Real component of SAR image
            self.p_xyz_i: Imaginary component of SAR image
        """
        return self.p_xyz_r, self.p_xyz_i

def _prep_inputs(radar_data, rx_offset, antenna_locs, use_4_rx):
    """
    Reformats inputs to prepare for input to GPU
    Parameters:
        - radar_data (numpy array): Radar measurement data starting as a (#Meas, #Sample/Meas, #RXAntenna) shaped array
        - rx_offset (list): 2D list of offsets from TX antenna to each RX antenna. Shape should be (#RXAntenna, 3)
        - antenna_locs: TX antenna locations of each measurement. Should be one location per measurement
        - use_4_rx: If the data contains 4 RX antennas 
    """
    if use_4_rx:
        # Reorder radar data to be (#Meas, #RXAntenna, #Sample/Meas)
        radar_data = np.transpose(radar_data, (0,2,1)) 
        # Repeat TX locations for each RX location, such that there is 1 TX location per (#Meas x #RXAntenna)
        antenna_locs = np.repeat(antenna_locs, radar_data.shape[1], axis=0)
    rx_offset = np.array(rx_offset) # Convert list to numpy array
    measurements = radar_data.reshape((-1)) # Flatten radar measurements into 1D array for cuda
    return rx_offset, measurements, antenna_locs

def _run_cuda_image(x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offset, slope, wavelength, bandwidth, num_samples, use_4_rx=False, is_ti_radar=False, use_interpolated_processing=False):
    '''
    Computes a SAR image on the GPU.
    Parameters: 
        - x_locs (numpy array): X locations to process SAR image at
        - y_locs (numpy array): Y locations to process SAR image at
        - z_locs (numpy array): Z locations to process SAR image at
        - antenna_locs (numpy array): Locations of TX antennas at each measurement
        - radar_data (numpy array): Radar measurement data shaped as (#Meas, #Sample/Meas, #RXAntenna) or (#Meas, #Sample/Meas)
        - rx_offset (numpy array): Offset from TX to each of the RX antennas. Shape should be (#RXAntenna, 3)
        - slope (float): Slope of the radar chirp
        - wavelength (float): Starting wavelength of the radar chirp
        - use_4_rx (bool): If the data contains 4 RX antennas
        - is_ti_radar (bool): If the data is real data coming from a TI radar (e.g., the 77GHz radar)
        - use_interpolated_processing (bool): Whether or not to use interpolated processing (e.g., backprojection) instead of a complete matched filter. Backprojection will be significantly faster in most scenarios
    '''
    # Find number of GPUs available
    num_gpus = drv.Device.count()
    print(f'Detected {num_gpus} CUDA Capable device(s)')

    # Prepare input to cuda
    norm_factor = antenna_locs.shape[0]
    samples_per_meas = radar_data.shape[1]
    rx_offset_flat, measurements, antenna_locs = _prep_inputs(radar_data, rx_offset, antenna_locs, use_4_rx)
    meas_real = np.ascontiguousarray(measurements.real)
    meas_imag = np.ascontiguousarray(measurements.imag)
    gc.collect()
    antenna_locs_flat = np.array(antenna_locs).flatten()
    num_x = len(x_locs)
    num_y = len(y_locs)
    num_z = len(z_locs)
    num_ant = len(antenna_locs)
    num_rx_ant = len(rx_offset)
    fft_spacing = np.float32(3e8/(2*bandwidth)*num_samples/(radar_data.shape[1]))

    # Define Grid size
    threads_per_block = 512
    grid_dim = int(num_x*num_y*num_z / threads_per_block / num_gpus) + 1
    print(f'Starting GPU computation')

    # Start a new thread for each GPU. Each thread is responsible for starting, stopping, and cleaning the CUDA code
    gpu_thread_list = []
    for i in range(num_gpus):
        gpu_thread = GPUThread(i, x_locs, y_locs, z_locs, antenna_locs_flat, meas_real, meas_imag, rx_offset_flat, slope, wavelength, fft_spacing, num_x, num_y, num_z, num_ant, num_rx_ant, samples_per_meas, is_ti_radar, use_interpolated_processing, threads_per_block, grid_dim, num_gpus)
        gpu_thread.start()
        gpu_thread_list.append(gpu_thread)

    # Get outputs from each GPU when they are done
    p_xyz_r = []
    p_xyz_i = []
    for thread in gpu_thread_list:
        thread.join()
        res_real, res_i = thread.get_res()
        p_xyz_r.append(res_real)
        p_xyz_i.append(res_i)

    # Sum outputs from all GPUs
    p_xyz_r = np.sum(p_xyz_r, axis=0)
    p_xyz_i = np.sum(p_xyz_i, axis=0)
    # Reshape into 3D arrays
    image_shape = (num_x, num_y, num_z)
    p_xyz_r = p_xyz_r.reshape((image_shape))
    p_xyz_i = p_xyz_i.reshape((image_shape))
    # Combine real/imaginary parts into a complex image
    p_xyz = p_xyz_r + 1j*p_xyz_i
    p_xyz /= norm_factor 

    del gpu_thread_list
    return p_xyz, norm_factor, antenna_locs


def image_cuda(x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offsets, slope, wavelength, bandwidth, num_samples, use_4_rx=False, is_ti_radar=False, use_interpolated_processing=False, return_phase_centers=True):
    '''
    Computes a SAR image on the GPU. 
    If needed, this function will break the processing into multiple separate calls to the GPU to limit the GPU memory of each call. 
    It will also parallelize across all available GPUs.
    Parameters: 
        - x_locs (numpy array): X locations to process SAR image at
        - y_locs (numpy array): Y locations to process SAR image at
        - z_locs (numpy array): Z locations to process SAR image at
        - antenna_locs (numpy array): Locations of TX antennas at each measurement
        - radar_data (numpy array): Radar measurement data shaped as (#Meas, #Sample/Meas, #RXAntenna) or (#Meas, #Sample/Meas)
        - rx_offset (numpy array): Offset from TX to each of the RX antennas. Shape should be (#RXAntenna, 3)
        - slope (float): Slope of the radar chirp
        - wavelength (float): Starting wavelength of the radar chirp
        - num_samples (float): Number of samples in the original radar chirp (note: not the number of samples after the FFT)
        - use_4_rx (bool): If the data contains 4 RX antennas
        - is_ti_radar (bool): If the data is real data coming from a TI radar (e.g., the 77GHz radar)
        - return_phase_centers (bool): Whether to compute and return the virtual phase centers (between TX and RX) of each measurement location
    '''
    sum_image = None
    if not use_interpolated_processing:
        sum_image, norm_factor, locs_rep = _run_cuda_image(x_locs, y_locs, z_locs, antenna_locs, radar_data, rx_offsets, slope, wavelength, bandwidth, num_samples, use_4_rx, is_ti_radar, use_interpolated_processing)
        sum_image /= norm_factor

        rx_locs = copy.deepcopy(locs_rep)
        rx_offsets = np.array(rx_offsets)
        rx_offset_rep = np.tile(rx_offsets, reps=(len(locs_rep)//len(rx_offsets),1))
        rx_locs += rx_offset_rep
        center_locs = (locs_rep + rx_locs)/2
    else:
        # If using the interpolated processing, the memory consumption will be higher. Therefore, break the call into multiple different groups as needed
        max_num_ant = 4096*32 # must be multiple of max_rx, reduce this to reduce amount of GPU memory used by each group
        num_rx = len(rx_offsets)
        num_ant_groups = math.ceil(antenna_locs.shape[0]*num_rx/max_num_ant)

        # Divide each radar location into different groups, compute the image for each group, and sum the final images
        for i in range(num_ant_groups):
            loc_subset = antenna_locs[i*max_num_ant//num_rx:(i+1)*max_num_ant//num_rx]
            meas_subset = radar_data[i*max_num_ant//num_rx:(i+1)*max_num_ant//num_rx]
            image_subset, norm_factor, locs_rep = _run_cuda_image(x_locs, y_locs, z_locs, loc_subset, meas_subset, rx_offsets, slope, wavelength, bandwidth, num_samples, use_4_rx, is_ti_radar, use_interpolated_processing)
            
            # Compute the virtual antenna centers (average of TX and RX location) for each measurement location
            if return_phase_centers:
                rx_locs = copy.deepcopy(locs_rep)
                rx_offsets = np.array(rx_offsets)
                rx_offset_rep = np.tile(rx_offsets, reps=(len(locs_rep)//len(rx_offsets),1))
                rx_locs += rx_offset_rep
                center_locs_subset = (locs_rep + rx_locs)/2

            if i == 0:
                sum_image = image_subset
                center_locs = center_locs_subset
            else:
                sum_image += image_subset
                center_locs = np.concatenate((center_locs, center_locs_subset), axis=0)

    if return_phase_centers:
        return sum_image, center_locs
    return sum_image

