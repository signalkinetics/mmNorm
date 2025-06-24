/*
This file contains functions for processing raw radar data into a mmWave image
*/

#include "imaging.h"

namespace py = pybind11;

Imaging::Imaging(std::vector<float> x_locs, std::vector<float> y_locs, std::vector<float> z_locs, std::string radar_type, std::string aperture_type, bool is_simulation) {
	/*
	Initialization for image processing

	Parameters:
		x_locs/y_locs/z_locs: Defines grid of locations for each voxel in desired mmWave image
		radar_type: Radar type to process (77_ghz or 24_ghz)
		aperture_type: Aperture type to process (normal or almost-square)
		is_simulation: True if processing a simulation image, False if processing real-world data
	*/
	this->x_locs = x_locs;
	this->y_locs = y_locs;
	this->z_locs = z_locs;   

	//Load radar parameters from JSON file
	auto proc_type = is_simulation ? "simulation" : "robot_collected";
    auto res = get_radar_params(radar_type, proc_type, aperture_type, "../utils/params.json");
    this->min_f = std::get<0>(res);
    this->max_f = std::get<1>(res);
    this->slope = std::get<2>(res);
    this->num_samples = std::get<3>(res);   
    this->num_rx_antenna = std::get<4>(res); 
	this->wavelength = SPEED_LIGHT / this->max_f;

}

Image Imaging::image(std::vector<Location> antenna_locs, std::vector<Measurement> measurements, std::vector<std::array<float, 3>> rx_offsets, float fft_spacing, bool apply_ti_offset, bool use_interpolated_processing) {
	/*
	Compute SAR image. This operation is parallelized across each slice in the x dimension

	Parameters:
		antenna_locs: Vector of locations where each measurement was taken from
		measurements: Vector of each radar measurement. If use_interpolated_processing is true, this should be FFTs of the radar measurements
		rx_offsets: Offsets from TX to each RX on the radar
		fft_spacing: spacing between successive samples in the range FFT (only used if use_interpolated_processing is True)
		apply_ti_offset: In practice, the TI radar has an ~15cm offset. If using the TI radar, apply an offset correction to improve image quality
		use_interpolated_processing:  Use a faster, but less accurate version of processing. 
			In this version, radar measurements are converted to range measurements using an FFT (before this function). 
			Then, we select the range bin closest to the round-trip distance for coorelation, instead of recomputing the exact coorelation each time.

	Returns:
		(X,Y,Z) shaped vector of processed image
	*/

	// Construct empty image
	Image p_xyz(x_locs.size(), std::vector<std::vector<std::complex<float>>>(y_locs.size(), std::vector<std::complex<float>>(z_locs.size())));

	// For progress bar
	const uint totalIterations = x_locs.size() * y_locs.size() * z_locs.size();
    uint completedIterations = 0;

	// Iterate through each voxel (over x/y/z)
	#pragma omp parallel for // This parallelizes across each x slice
	for (uint x = 0; x < x_locs.size(); x++) {
		float x_loc = x_locs[x];
		for (uint y = 0; y < y_locs.size(); y++) {
			float y_loc = y_locs[y];
			for (uint z = 0; z < z_locs.size(); z++) {
				float z_loc = z_locs[z];

				// For this voxel, compute a sum across every measurement
				std::complex<float> sum = std::complex<float>(0,0);
				uint used_antennas = 0;
				for (uint i = 0; i < antenna_locs.size(); i++) {
					used_antennas++;

					// Find the round-trip distance from TX-> voxel -> RX
					Location antenna_loc = antenna_locs.at(i);
					float antenna_x_diff = x_loc - antenna_loc[0];
					float antenna_y_diff = y_loc - antenna_loc[1];
					float antenna_z_diff = z_loc - antenna_loc[2];
					float forward_dist = std::sqrt(antenna_x_diff * antenna_x_diff + 
												   antenna_y_diff * antenna_y_diff + 
												   antenna_z_diff * antenna_z_diff);
					if (apply_ti_offset){ // This is an offset correction for the TI radar
						forward_dist += 0.15;
					}
					for (uint k = 0; k < this->num_rx_antenna; k++) {
						float rx_offset_x = rx_offsets[k][0];
						float rx_offset_y = rx_offsets[k][1];
						float rx_offset_z = rx_offsets[k][2];
						float rx_x_diff = x_loc - (antenna_loc[0] + rx_offset_x);
						float rx_y_diff = y_loc - (antenna_loc[1] + rx_offset_y);
						float rx_z_diff = z_loc - (antenna_loc[2] + rx_offset_z);
						float back_dist = std::sqrt(rx_x_diff * rx_x_diff + rx_y_diff * rx_y_diff + rx_z_diff * rx_z_diff);
						float distance = forward_dist + back_dist;



						if (use_interpolated_processing ){ // Apply faster but approximate coorelation
							// Check that our distance is valid
							if (distance < 0 || distance > fft_spacing*this->num_samples) {
								continue;
							}

							// Find which bin within the range FFT this distance falls
							int dist_bin = floorf(distance / fft_spacing/2);

							// Select the appropriate measurement, and coorelate with the AoA phase
							sum += measurements[i][dist_bin][k] * std::exp(std::complex<float>(0., -2. * M_PI * distance / this->wavelength));
						} else { // Apply more exact but slower coorelation
							for (uint j = 0; j < this->num_samples; j++) {
								sum += measurements[i][j][k] * std::exp(std::complex<float>(0., -2. * M_PI * distance * j * this->slope / SPEED_LIGHT)) * 
									std::exp(std::complex<float>(0., -2. * M_PI * distance / this->wavelength));
							}
						}
					}

				}
				// Update image with the sum (normalized by the number of measurements)
				p_xyz[x][y][z] = sum / std::complex<float>(used_antennas,0);

				// Progress bar
				#pragma omp critical
				{
					completedIterations++;
					float progress = static_cast<float>(completedIterations) / totalIterations * 100;
					std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << progress << "%";
					std::cout.flush();
				}

			}

		}

	}  

    std::cout << "\rProgress: 100.00%" << std::endl;
	return p_xyz;

}

// Use Pybind11 to make these functions accessible from python
PYBIND11_MODULE(imaging, m) {  
	py::class_<Imaging>(m,"Imaging")
		.def(py::init<std::vector<float>, std::vector<float>, std::vector<float>, std::string, std::string, bool>())
		.def("image", &Imaging::image);
}
