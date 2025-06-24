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