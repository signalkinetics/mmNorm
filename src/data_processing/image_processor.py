import numpy as np
import matplotlib.pyplot as plt
import time 
from scipy import interpolate
from scipy.constants import c
import gc

from utils import *
from utils import utilities
if utilities.load_param_json()['processing']['use_cuda']:
    from data_processing.cuda import py_image_gpu
    from data_processing.cuda import py_sn_est_gpu
else:
    from data_processing.cpp.imaging import Imaging

class ImageProcessor:

    def __init__(self, is_optitrack=False, radar_type='77_ghz'):
        self.is_optitrack = is_optitrack
        self.radar_type = radar_type

    def interpolate_measurements(self, radar_data, FFT_SIZE=4096):
        """ Interpolate the measurements and take FFT. 
        Parameters:
            - radar_data (np array): Array of radar measurements. Expecting shape (Num meas, Num samples, Num RX Antennas)
            - FFT_SIZE (int): Length of final FFT to interpolate radar data at
        Returns:
            - measurements_padded (np array): Interpolated FFT data. Will have shape (Num meas, FFT Size, Num RX Antenna)
        """
        assert radar_data.shape[1] <= FFT_SIZE
        measurements_padded = np.zeros((radar_data.shape[0], FFT_SIZE, radar_data.shape[2]), dtype=np.complex64)
        measurements_padded[:, :radar_data.shape[1], :] = radar_data
        interpolated_data = np.fft.fft(measurements_padded, axis=1)
        gc.collect() # Clean up from FFT operations
        return interpolated_data

    def get_interpolated_funct(self, transform_times, transforms):
        """ Create function which can interpolate between multi-dimensional transforms. 
        Parameters: 
            transform_times (np array): times that transforms was sampled at.
            transforms (np array): multi-dimensional transforms
        Returns:
            all_functs: a list of functions which can interpolate each dimension of the provided transform
        """
        all_functs = []
        for i in range(transforms.shape[1]):
            interp_funct = interpolate.interp1d(transform_times, transforms[:, i])
            all_functs.append(interp_funct)
        return all_functs

    def get_interpolated_transforms_from_funct(self, functs, dist_times):
        """ Interpolate transforms at given times
        Parameters:
            functs: list of interpolation functions
            dist_times: times to interpolate at
        Returns: 
            interpolated_transforms: transforms interpolated at provided times
        """
        interpolated_transforms = np.zeros((len(dist_times), len(functs)))
        for i in range(len(functs)):
            interpolated_transforms[:, i] = functs[i](dist_times)
        return interpolated_transforms

    def convert_opt_frame(self, loc):
        """
        Parameters:
            loc (numpy array): the location of the object as detected by the optitrack system

        Returns:
            Adjusted coordinate frame from optitrack to robot (data from "antenna" files)
        """
        return np.array([-loc[0], loc[2], loc[1]])

    def correlate_locs_and_meas(self, radar_type, robot_data, radar_data, exp_data, robot_loc_data=None, speed="speed_8"):
        """
        Find the robot locations where each radar measurement was taken by interpolating the robot trajectory.
        Only used in real world experiments.
        Parameters:
            radar_type (str): type of radar to use ("24_ghz, 77_ghz")
            robot_data (dict): loaded robot pickle file using the load_robot_file function
            radar_data (dict): loaded radar adc file using the load_radar_files function
            robot_loc_data (dict): loaded robot location data using the load_robot_loc_files function
            speed (str): the speed of the radar ("speed_4", "speed_8")

        Returns:
            Robot locations and radar data, with one-to-one correspondences
        """
        all_locs = []
        all_radar_data = []
        all_index = []
        assert radar_type in ['77_ghz', '24_ghz'], "Please choose a valid radar type."

        aperture_type = exp_data['aperture size']

        # Load imaging parameters
        params_dict = utilities.get_radar_parameters(radar_type=radar_type, is_sim=False, aperture_type=aperture_type)
        FREQUENCY = params_dict['min_f']
        NUM_FRAMES = params_dict['num_frames']
        SWEEP_TIME = params_dict['sweep_time']
        PERIODICITY = params_dict['periodicity']
        SAMPLES_PER_CHIRP = params_dict['num_samples']
        NUM_CHIRP = params_dict['num_chirps']
        wavelength = c / FREQUENCY
        repeated_meas_threshold = wavelength * (3 / 16)

        # Create interpolation functions for robot locations
        radar_type_num = radar_type[:2]
        robot_loc_data[f'times_{radar_type_num}'] = robot_loc_data[f'times_{radar_type_num}']
        interp_functs = self.get_interpolated_funct(robot_loc_data[f'times_{radar_type_num}'], robot_loc_data[f'tx_{radar_type_num}_locs'])

        # Find robot location for every radar measurement
        for i, filename in enumerate(robot_data['all_radar_filenames']):
            # Load radar file for this row
            created_ts = robot_data['all_radar_created_ts'][i]
            if created_ts == -1: continue
            if int(filename) not in radar_data:
                print(f"Couldn't find file {int(filename)}")
                continue

            # Each radar file contains NUM_FRAMES different radar measurements
            data = radar_data[int(filename)]
            current_row_locs = []
            for j in range(NUM_FRAMES):
                # Compute timestamp of jth measurement inside a file
                start = created_ts + j * (SWEEP_TIME + PERIODICITY)

                # Try to interpolate robot location at this measurement
                try:
                    loc = self.get_interpolated_transforms_from_funct(interp_functs, [start])[0]
                    # Remove two measurements that are too close together
                    if j != 0 and np.min(np.linalg.norm(np.array(current_row_locs)[:, :3] - loc[:3], axis=1)) < repeated_meas_threshold: 
                        continue
                except:
                    continue

                # Load radar data for this measurement
                new_data = data[j * SAMPLES_PER_CHIRP * NUM_CHIRP :
                                j * SAMPLES_PER_CHIRP * NUM_CHIRP + SAMPLES_PER_CHIRP, :]
                if ((radar_type == "24_ghz" and new_data.shape != (SAMPLES_PER_CHIRP, 2)) or
                    (radar_type == "77_ghz" and new_data.shape != (SAMPLES_PER_CHIRP, 4))):
                    print(f'File {filename} did not have expected number of frames. Skipping')
                    continue
                
                # Save data / location / index
                if self.is_optitrack: loc = self.convert_opt_frame(loc)
                all_locs.append(loc)
                current_row_locs.append(loc)
                all_radar_data.append(new_data)
                all_index.append(i * NUM_FRAMES + j)

        # Return all data
        all_locs = np.array(all_locs)
        data = {'radar_data': all_radar_data, 'poses': all_locs}
        return data

    def generate_sar_image_radar_frame(self, radar_type, data, exp_data, data_bg=None, plot=False, bg_chirp_num=0, ):
        """
        This function computes a standard SAR (Synthetic Aperture Radar) image based on the loaded robot data and radar data.
        It does not compute surface normals, please use generate_mmwave_surface_normals for that functionality. 

        Parameters:
            radar_type (str): type of radar to use ("24_ghz, 77_ghz")
            data (dict): data loaded by GenericLoader
            data_bg (dict): background data to be subtracted. Use only when processing 24_ghz.
            plot (bool): if True, plots the radar locations and imaging locations for debugging purposes.
        
        Returns:
            tuple of:
                numpy array of image data
                tuple of numpy arrays (x_locs, y_locs, z_locs)
                numpy array of radar frame locations
        """
        # Load data and radar parameters
        aperture_type = exp_data['aperture size']
        rx_offset_dir = exp_data['rx offset']
        is_tripod = exp_data['background'] == 'tripod'
        print(f'aperture type: {aperture_type} rx offset: {rx_offset_dir} is_Tripod: {is_tripod}')

        locs = data['poses']
        location0 = [ 0.42,  -0.23,  0.23]
        radar_data = np.array(data['radar_data'])
        params_dict = utilities.get_radar_parameters(radar_type=radar_type, is_sim=False, aperture_type=aperture_type)
        FREQUENCY = params_dict['min_f']
        SAMPLES_PER_CHIRP = params_dict['num_samples']
        wavelength = c / FREQUENCY
        slope = params_dict['slope']
        bandwidth = params_dict['bandwidth']

        # If background data is provided, subtract the background data from each radar measurement
        if data_bg is not None:
            bg_meas = np.array(data_bg['radar_data'])[bg_chirp_num, :SAMPLES_PER_CHIRP, :]
            for i in range(len(radar_data)):
                radar_data[i, :SAMPLES_PER_CHIRP, :] -= bg_meas

        # Convert all locations into radar frame
        locs_radar_frame = []
        for loc in locs:
            if self.is_optitrack: loc[:3] -= np.array([-1.33640909, -0.87117398, 1.17517328])
            else: loc[:3] -= location0
            locs_radar_frame.append(loc)
        locs_radar_frame = np.array(locs_radar_frame)

        # Define bounds to image within
        min_locs_radar_frame = [np.min(locs_radar_frame[:, 0]), np.min(locs_radar_frame[:, 1]), np.min(locs_radar_frame[:, 2])]
        max_locs_radar_frame = [np.max(locs_radar_frame[:, 0]), np.max(locs_radar_frame[:, 1]), np.max(locs_radar_frame[:, 2])]
        radar_min_x = min_locs_radar_frame[0]
        radar_max_x = max_locs_radar_frame[0]
        radar_min_y = min_locs_radar_frame[1]
        radar_max_y = max_locs_radar_frame[1]
        if radar_type =='77_ghz':
            min_x = -0.1
            max_x = 0.45
            min_y = -0.05
            max_y = 0.55
        else:
            min_x = 0 - 0.05
            max_x = radar_max_x + 0.05
            min_y = 0.05 
            max_y = 0.65
        if is_tripod:
            min_z, max_z = -0.37, -0.15
        else:
            min_z, max_z = -0.55, -0.3

        # Imaging resolution in each dimension
        step_size_x = 0.001 
        step_size_y = 0.001 
        step_size_z = 0.001
        
        # Define locations of each voxel
        x_locs = np.arange(min_x, max_x, step_size_x)
        y_locs = np.arange(min_y, max_y, step_size_y)
        z_locs = np.arange(min_z, max_z, step_size_z)

        # Plot the locations for debugging
        if plot:
            plt.scatter(locs_radar_frame[:, 0], locs_radar_frame[:, 1], c="blue") # Where the antennas are
            plt.scatter(x_locs, np.arange(0, len(x_locs)) * (max_y - min_y) / len(x_locs) + min_y, c="red") # Where we are performing calculations
            plt.xlabel('X')
            plt.ylabel('Y')
            min_imaging_locs = [np.min(x_locs), np.min(y_locs), np.min(z_locs)]
            max_imaging_locs = [np.max(x_locs), np.max(y_locs), np.max(z_locs)]
            plt.xlim((min(min_imaging_locs[0], min_locs_radar_frame[0]) - 0.03, max(max_imaging_locs[0], max_locs_radar_frame[0]) + 0.03))
            plt.ylim((min(min_imaging_locs[1], min_locs_radar_frame[1]) - 0.03, max(max_imaging_locs[1], max_locs_radar_frame[1]) + 0.03))
            plt.show()

        # Define imaging parameters
        if radar_type == "24_ghz":
            rx_offset = -0.021
            rx_offsets = [[0,rx_offset, 0],
                         [0,rx_offset-3/240/2, 0]]
            max_rx = 2
        elif radar_type == "77_ghz":
            tx_offset = 0.005 
            rx_offset = wavelength / 2
            if rx_offset_dir == 'x':
                rx_offsets = [[-tx_offset - rx_offset * 3, 0, 0], 
                                [-tx_offset - rx_offset * 2, 0, 0], 
                                [-tx_offset - rx_offset,     0, 0], 
                                [-tx_offset,                 0, 0]]
            elif rx_offset_dir == 'y':
                rx_offsets = [[0, -tx_offset - rx_offset * 3, 0], 
                              [0, -tx_offset - rx_offset * 2, 0], 
                              [0, -tx_offset - rx_offset,     0], 
                              [0, -tx_offset,                 0]]
            else:
                raise Exception('rx_offset_dir must be x or y')
            max_rx = 4

        start = time.time()
        apply_ti_offset = radar_type=='77_ghz'

        # Interpolate measurements if applicable
        radar_data_clip = radar_data[:,:SAMPLES_PER_CHIRP, :max_rx]
        use_interpolated_processing = utilities.load_param_json()['processing']['use_interpolated_processing']
        if use_interpolated_processing: # Process on GPU
            measurement_data = self.interpolate_measurements(radar_data_clip)
        else:
            measurement_data = radar_data_clip

        # Compute image (on CPU or GPU)
        print(f'Starting processing')
        if utilities.load_param_json()['processing']['use_cuda']: # Process on GPU
            radar_data_clip = radar_data[:,:SAMPLES_PER_CHIRP, :max_rx]
            image = py_image_gpu.image_cuda(x_locs, y_locs, z_locs, locs_radar_frame[:,:3], measurement_data, np.array(rx_offsets), slope, wavelength, bandwidth, SAMPLES_PER_CHIRP, use_4_rx=True if max_rx!=1 else False, is_ti_radar=apply_ti_offset, use_interpolated_processing=use_interpolated_processing)
        else: # Process on CPU
            fft_spacing = np.float32(3e8/(2*bandwidth)*SAMPLES_PER_CHIRP/(measurement_data.shape[1]))
            imaging = Imaging(x_locs, y_locs, z_locs, radar_type, aperture_type, False)
            image = np.array(imaging.image(locs_radar_frame[:,:3], measurement_data, rx_offsets, fft_spacing, apply_ti_offset, use_interpolated_processing))
        image /= len(locs)
        end = time.time()
        print(f"took {(end-start)} to process image")

        return image, (x_locs, y_locs, z_locs), locs_radar_frame

    def generate_mmwave_surface_normals(self, radar_type, data, exp_data, is_sim=False):
        """
        Generate mmWave surface normal estimates. This also generates a standard SAR image in the process, and returns that too. 

        Parameters:
            radar_type (str): type of radar to use ("10_ghz, 24_ghz, 77_ghz"). Note: Only 77_ghz is supported at the moment.
            data (dict): data loaded by GenericLoader
            exp_data (dict): Data describing experiment, loaded by GenericLoader
            is_sim (bool): Whether data is simulated or not
        
        Returns:
            tuple of:
                numpy array of SAR image data
                tuple of numpy arrays (x_locs, y_locs, z_locs)
                numpy array of mmWave surface normal estimates
                numpy array of locations of each radar measurement (as virtual antenna center between TX and RX)
                numpy array of raw radar measurement data
        """

        # Load data and radar parameters
        aperture_type = exp_data['aperture size']

        locs = data['poses']
        location0 = [ 0.42,  -0.23,  0.23]
        radar_data = np.array(data['radar_data'])
        params_dict = utilities.get_radar_parameters(radar_type=radar_type, is_sim=False, aperture_type=aperture_type)
        FREQUENCY = params_dict['min_f']
        SAMPLES_PER_CHIRP = params_dict['num_samples']
        wavelength = c / FREQUENCY
        slope = params_dict['slope']
        bandwidth = params_dict['bandwidth']
        

        # Define bounds/resolution of image
        step_size_x = 0.001 
        step_size_y = 0.001 
        step_size_z = 0.001
        min_x = -0.1
        max_x = 0.45
        min_y = -0.05
        max_y = 0.55
        min_z, max_z = -0.5, -0.25


        # Convert all locations into radar frame
        all_locs = []
        all_meas = []
        max_rx = 1 if is_sim else 4
        for i, loc in enumerate(locs):
            if self.is_optitrack: loc_radar_frame = loc - np.array([-1.33640909, -0.87117398, 1.17517328])
            else: loc_radar_frame = loc[:3] - location0 #locs[0]
            all_locs.append(loc_radar_frame)
            all_meas.append(radar_data[i,:SAMPLES_PER_CHIRP, :max_rx])
        all_locs = np.array(all_locs)
        all_meas = np.array(all_meas)
            
        # Create imaging locations
        x_locs = np.arange(min_x, max_x, step_size_x)
        y_locs = np.arange(min_y, max_y, step_size_y)
        z_locs = np.arange(min_z, max_z, step_size_z)

        # Define radar parameters
        if not is_sim:
            slope = 58.6116 * 1e6 * 1e6 / 10e6 #MHz/us *1e6us/s * 1/10e6 sec/samp * 1e6 Hz/MHz  = HZ/samp 
            tx_offset = 0.005 
            rx_offset = wavelength / 2
            if aperture_type=='original':
                rx_offsets = [[0, -tx_offset - rx_offset * 3, 0], 
                                [0, -tx_offset - rx_offset * 2, 0], 
                                [0, -tx_offset - rx_offset,     0], 
                                [0, -tx_offset,                 0]]
            elif aperture_type=='almost-square':
                rx_offsets = [[-tx_offset - rx_offset * 3, 0, 0], 
                                [-tx_offset - rx_offset * 2, 0, 0], 
                                [-tx_offset - rx_offset,     0, 0], 
                                [-tx_offset,                 0, 0]]
        else:
            rx_offsets = [[0,0,0]]
            SPEED_LIGHT=2.99792458e8
            max_f = params["max_f"]
            min_f = params["min_f"]
            slope = (max_f - min_f) / (params["num_samples"]-1)
            wavelength = c/min_f
        apply_ti_offset = radar_type=='77_ghz'

        # Interpolate measurements if applicable
        radar_data_clip = radar_data[:,:SAMPLES_PER_CHIRP, :max_rx]
        use_interpolated_processing = utilities.load_param_json()['processing']['use_interpolated_processing']
        if use_interpolated_processing: # Process on GPU
            measurement_data = self.interpolate_measurements(radar_data_clip)
        else:
            measurement_data = radar_data_clip

        # Compute SAR image on GPU
        sum_image, center_locs = py_image_gpu.image_cuda(x_locs, y_locs, z_locs, all_locs, measurement_data, np.array(rx_offsets), slope, wavelength, bandwidth, SAMPLES_PER_CHIRP, use_4_rx=True, is_ti_radar=not is_sim,use_interpolated_processing=use_interpolated_processing)

        # Compute normal estimates on GPU
        normals = py_sn_est_gpu.normal_estimates_cuda(sum_image, x_locs, y_locs, z_locs, all_locs, measurement_data, np.array(rx_offsets), slope, wavelength, bandwidth, SAMPLES_PER_CHIRP, use_4_rx=True,is_ti_radar=not is_sim,use_interpolated_processing=use_interpolated_processing)
                
        # Normalize normals so they are unit vectors
        normals /=  np.linalg.norm(normals, axis=-1)[...,np.newaxis].repeat(3, axis=-1)

        return sum_image, (x_locs, y_locs, z_locs), normals, center_locs, all_meas

 
 
 
 

