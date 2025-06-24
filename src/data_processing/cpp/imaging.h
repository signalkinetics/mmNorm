#include "mmwave_common.h"


class Imaging {
	public:
		std::vector<float> x_locs;
		std::vector<float> y_locs;
		std::vector<float> z_locs;

		float wavelength;
		float slope;
		float min_f;
		float max_f;
		uint num_rx_antenna;
		uint num_samples;

		Imaging(std::vector<float>, std::vector<float>, std::vector<float>, std::string, std::string, bool);

		Image image(std::vector<Location> antenna_locs, std::vector<Measurement>, std::vector<std::array<float, 3>>, float, bool = false, bool= false);

};
