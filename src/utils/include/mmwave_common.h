/*
This file includes C++ utility functions
*/

#include "json.hpp"
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <iostream>


#define SPEED_LIGHT 2.99792458e8

using Location = std::array<float, 3>;
using Measurement = std::vector<std::vector<std::complex<float>>>; // Number of RX x Number of Samples
using Image = std::vector<std::vector<std::vector<std::complex<float>>>>; // X x Y x Z complex vector


std::tuple<float, float, float, uint, uint> get_radar_params(std::string radar_type, std::string data_type, std::string aperture_type, std::string path){
    /*
    Load the parameters of a given radar from the experiment json
    Parameters:
        radar_type (string): The radar type to load parameters for (77_ghz or 24_ghz)
        data_type (string): simulation or robot_collected
        aperture_type (string): normal or almost-square
        path (string): Path to json file
    */
    
    // Load JSON file
    std::ifstream f(path);
    nlohmann::json data = nlohmann::json::parse(f);

    // Parse information from JSON
    float max_f = 0;
    float min_f = 0;
    uint num_samples = 0;
    float slope = 0;
    uint num_rx_antenna = 0;
    if (data_type == "robot_collected") {
        max_f = data[data_type][radar_type][aperture_type]["max_f"];
        min_f = data[data_type][radar_type][aperture_type]["min_f"];
        num_samples = data[data_type][radar_type][aperture_type]["num_samples"];
        slope = data[data_type][radar_type][aperture_type]["slope"]; 
        num_rx_antenna = data[data_type][radar_type][aperture_type]["num_rx_antenna"]; 
    }
    else {
        max_f = data[data_type][radar_type]["max_f"];
        min_f = data[data_type][radar_type]["min_f"];
        num_samples = data[data_type][radar_type]["num_samples"];
        slope = data[data_type][radar_type]["slope"]; 
        num_rx_antenna = data[data_type][radar_type]["num_rx_antenna"]; 
    }

    // Return as a tuple
    return {min_f, max_f, slope, num_samples, num_rx_antenna};
}