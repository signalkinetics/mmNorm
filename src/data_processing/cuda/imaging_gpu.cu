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
}

int main() {
    return 0;
}