"""
This file is responsible for loading and saving data from the dataset
It manages the proper folder structure for data in the dataset
"""

import os
import pickle
import numpy as np
from PIL import Image
from utils import *
import json

from utils import utilities
from utils.object_information import *

class GenericLoader:

    def __init__(self, obj_id: str, name: str, is_sim=True, is_los=True, exp_num=1):
        """
        Initializes GenericLoader for loading data from the dataset

        Parameters:
            obj_id (str): id of the object (e.g. "001"). Note: This ID is expected to be the full 3 digit long ID, including leading zeros.
            name (str): name of the object (e.g. "phillips_screw_driver"). Note: This name is expected to match the official name in the YCB dataset (including underscores, not spaces)
            is_sim (bool): True if loading or saving data from simulation environment
            is_los (bool): True if line-of-sight experiment. False for non-line-of-sight. This parameter is ignored for simulation data
            exp_num (int): Experiment number. None or 'None' will default to 1. This is ignored if using simulation
        """
        self.obj_id = obj_id
        self.name = name
        self.is_sim = is_sim
        self.is_los = is_los
        if exp_num == 'None' or exp_num is None: exp_num = 1
        self.exp_num = exp_num

    def _get_path(self, radar_type=None, x_angle=0, y_angle=0, z_angle=0, is_processed=False, is_json=False):
        """
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz","camera"). If left blank,
                only the path until the specified angle will be returned.
            angle (int): rotation angle in degrees of the loading object
            is_processed (bool): if True, will point to the processed folder
            is_json (bool): if True, return experiment json path

        Returns:
            A string that is the path to the directory containing the data for the specific radar type
        """
        dir_path = utilities.get_root_path()
        obj_path = self.obj_id + '_' + self.name
        sim_path = "simulation" if self.is_sim else "robot_collected"
        angle_path = str(x_angle) + "_" + str(y_angle) + "_" + str(z_angle)
        los_path = "" if self.is_sim or self.is_los is None else ("los" if self.is_los else "nlos")
        processed_path = "processed" if is_processed else "unprocessed"
        path_to_dataset = utilities.load_param_json()['processing']['path_to_data']
        path_to_processed = f"{dir_path}/{path_to_dataset}/{obj_path}/{sim_path}/{angle_path}/{los_path}/{processed_path}"
        exp_path = f'exp{self.exp_num}' if not self.is_sim else ''
        if radar_type is None:
            raise ValueError("You should not use Radar_type = None")
            return path_to_processed 
        elif is_json:
            return f"{dir_path}/{path_to_dataset}/{obj_path}/{sim_path}/{angle_path}/{exp_path}"
        elif radar_type == "camera":
            camera_path = f"{dir_path}/{path_to_dataset}/{obj_path}/{sim_path}/{angle_path}/{exp_path}/{los_path}/{processed_path}/camera"
            return camera_path
        else:
            path_to_processed = f"{dir_path}/{path_to_dataset}/{obj_path}/{sim_path}/{angle_path}/{exp_path}/{los_path}/{processed_path}/radars"
            return f"{path_to_processed}/{radar_type}"

    def _find_obj_angles(self):
        """
        Finds the object angles for the given experiment number. 
        This function will iterate through each angle folder to find the matching exp folder. 
        This assumes the matching exp data has been downloaded from AWS.

        Returns:
            a list of integers cooresponding to the X, Y, Z angles of the object for this experiment
        """
        if self.is_sim: return None
        obj_path = self.obj_id + '_' + self.name
        path_to_dataset = utilities.load_param_json()['processing']['path_to_data']
        data_folder = f'{utilities.get_root_path()}/{path_to_dataset}/{obj_path}/robot_collected'
        for angle_folder in os.listdir(data_folder):
            for exp_folder in os.listdir(f'{data_folder}/{angle_folder}'):
                if exp_folder == f'exp{self.exp_num}': 
                    angles = [int(angle) for angle in angle_folder.split('_')] # Convert folder name to list of 3 angles
                    return angles 
        raise Exception(f"Couldn't find any folder matching exp{self.exp_num} for object {self.obj_id}_{self.name}")
    
    def _parse_angles(self, angles):
        """ Parse angles into 3 separate values. 
        If angles is None, and using simulation, hard code x/y/z to 0,0,0
        If angles is None, and not using simulation, determine the angles based on the experiment number
        
        Parameters: 
            angles (tuple or None): Tuple of angles or None
        Returns: Tuple of angles (3 integers)
        """
        if angles is None:
            if self.is_sim:
                x_angle, y_angle, z_angle = (0, 0, 0) # Default to 0,0,0 for simulation data
            else:
                x_angle, y_angle, z_angle = self._find_obj_angles()
        else:
            x_angle, y_angle, z_angle = angles
        return x_angle, y_angle, z_angle

    def get_path_to_mesh(self, obj=False):
        """
        Parameters:
            obj (bool): if True, use obj file instead of stl file

        Returns:
            A string that is the path to the .obj or .stl file that contains the original mesh file provided by YCB
        """ 
        assert self.is_sim , "Mesh files are only used for simulation. Use GenricLoader with is_sim=True"
        
        dir_path = utilities.get_root_path() 
        path_to_dataset = utilities.load_param_json()['processing']['path_to_data']
        mesh_path = f"{dir_path}/{path_to_dataset}/{self.obj_id}_{self.name}"
        return f"{mesh_path}/textured.obj" if obj else f"{mesh_path}/nontextured.stl"

    def get_path_to_uniform_mesh(self, num_vert=None):
        """
        This function returns the path to a uniform mesh. 
        The uniform mesh is the YCB mesh that has been re-sampled to have roughly uniform spacing of vertices across the mesh.
        Uniform meshes can be created with different number of vertices 

        Parameters: 
            num_vert (int): Number of vertices in uniform mesh
        Returns:
            A string that is the path to the uniform mesh file with the given number of vertices. 
        """
        assert self.is_sim , "Mesh files are only used for simulation. Use GenricLoader with is_sim=True"
        
        dir_path = utilities.get_root_path()
        path_to_dataset = utilities.load_param_json()['processing']['path_to_data']
        pcd_path = f"{dir_path}/{path_to_dataset}/{self.obj_id}_{self.name}"
        ext = '' if num_vert is None else f'_{num_vert}'
        return f"{pcd_path}/uniform{ext}.stl"

    def save_image(self, radar_type, image, x_locs, y_locs, z_locs, angles=None, antenna_locs=None, ext= ""):
        """
        Saves processed image to the specified directory

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            image (numpy array): processed image by ImageProcessor
            x_locs (numpy array): sampled x locations of image
            y_locs (numpy array): sampled y locations of image
            z_locs (numpy array): sampled z locations of image
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world   
            antenna_locs (numpy array): antenna locations used to processed image
            ext (str): extension of saved file name
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        data = {"image": image, "x_locs": x_locs, "y_locs": y_locs, "z_locs": z_locs, "antenna_locs": np.array(antenna_locs)}
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + "/processed_image"+ext+".pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_raw_sim_files(self, radar_type, channels, radar_shifted_locs, angles, ext=""):
        """
        Saves raw simulation wave files to the specified directory

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            channels (list(list)): wave channel information from simulation
            radar_shifted_locations (list(list)): radar shifted locations information from simulation
            angle (int): rotation angle in degrees of the loading object
        """
        x_angle, y_angle, z_angle = angles
        data = {"channels": channels, "radar_shifted_locs": radar_shifted_locs}
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False) + f"/raw_sim_files{ext}.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
    def load_raw_sim_files(self, radar_type, angles, ext=""):
        """
        loads raw simulation wave files 

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object
        """
        x_angle, y_angle, z_angle = angles
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False) + f"/raw_sim_files{ext}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
        
    def load_image_file(self, radar_type, angles=None, background_subtraction=None, ext="", crop=False, crop_high=False):
        """
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world  

        Returns: 
            Loaded processed image pickle file generated by ImageProcessor
        """
        # Find object angles
        x_angle, y_angle, z_angle = self._parse_angles(angles)

        # Load image
        if self.is_sim: background_subtraction = None
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + f"/processed_image{ext}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        image = data["image"]
        x_locs = data["x_locs"]
        y_locs = data["y_locs"]
        z_locs = data["z_locs"]
        antenna_locs = np.array(data["antenna_locs"])

        # Crop (vertically) if necessary
        if crop:
            if radar_type=='77_ghz' and not self.is_sim:
                if not crop_high:
                    image = image[:,:,12:]
                    z_locs = z_locs[12:]
                    image = image[:,:,:-7]
                    z_locs = z_locs[:-7]
                else:
                    image = image[:,:,17:]
                    z_locs = z_locs[17:]
                    image = image[:,:,:-3]
                    z_locs = z_locs[:-3]
            else:
                image = image[:,:,:-3]
                z_locs = z_locs[:-3]

        # Apply background subtraction if necessary (only used for 24GHz radar)
        if background_subtraction is not None and radar_type=='24_ghz':
            # Load empty image. All empty images are saved with a unique ID and the name "EMPTY"
            bg_loader = GenericLoader(background_subtraction, 'EMPTY', is_sim=False, is_los=True, exp_num=self.exp_num)
            bg_image, (bg_x_locs, bg_y_locs, bg_z_locs), _ = bg_loader.load_image_file('24_ghz', (0,0,0), ext=ext, crop=crop, crop_high=crop_high)
            
            # Crop background image until shape matches image shape & apply background subtraction
            if bg_image.shape[2] != image.shape[2]:
                diff = image.shape[2] -  bg_image.shape[2]
                image = image[:,:,diff:]
            extra =  np.array(bg_image.shape) - np.array(image.shape)
            if extra[0] != 0:
                image =np.abs(image) - np.abs( bg_image[:-extra[0]])
            else:
                image= np.abs(image) - np.abs(bg_image)
        return image, (x_locs, y_locs, z_locs), antenna_locs
    
    def load_camera_data(self):
        """
        Load data taken from the camera. This will load a dictionary including the RGB-D data and other information

        Returns: Dictionary of camera data
        """
        x, y, z = self._find_obj_angles()
        path = self._get_path(radar_type="camera", is_processed=False, x_angle=x, y_angle=y, z_angle=z)
        if not os.path.exists(path):
            print(f'Couldnt find object: {path}')
            return None
        filenames = sorted(os.listdir(path))
        for filename in filenames:
            if filename[-4:] != '.pkl' or filename[:10] != 'continuous':
                continue
            with open(f'{path}/{filename}', 'rb') as f:
                cam_data = pickle.load(f)
            break

        return cam_data

    def load_camera_masks(self, ext=''):
        """
        Load masks of RGB data generated using SAM. These masks are human generated and serve as ground truth
        
        Parameters:
            ext (string): extension of file to include
        Returns:
            Dictionary of all data saved by mask generation tool
        """
        x, y, z = self._find_obj_angles()
        path = self._get_path(radar_type="camera", is_processed=True, x_angle=x, y_angle=y, z_angle=z)
        path = f'{path}/camera_masks{ext}.pkl'
        if not os.path.exists(path):
            print(f'Couldnt find object: {path}')
            return None
        with open(f'{path}', 'rb') as f:
            masks = pickle.load(f)
        return masks
        
    def save_camera_masks(self, rgbd_mask, depth_masked, xyz_masked, cam_data, ext=''):
        """
        Saves masks of RGB data generated using SAM. These masks are human generated and serve as ground truth
        
        Parameters:
           rgbd_mask: Mask from RGB image. Numpy array WxH
           depth_masked: Masked depth image. This has been inpainted to fill in small holes. Numpy array WxH
           xyz_masked: Masked depth image converted to XYZ points (in camera frame). Numpy array WxHx3
           cam_data: Original camera data (will only save pose from this data)
        """
        path = self._get_path(radar_type="camera", is_processed=True)
        masks_path = f'{path}/camera_masks{ext}.pkl'
        os.makedirs(os.path.dirname(masks_path), exist_ok=True)
        data = {
            'rgbd_mask': rgbd_mask,
            'masked_depth': depth_masked,
            'pose': cam_data['pose'],
            'masked_xyz': xyz_masked,
        }
        with open(masks_path, "wb") as f:
            pickle.dump(data, f)
               
    def save_radar_masks(self, mask, radar_type='77_ghz', ext='', angles=None):
        """
        Save masks on 2D radar image
        
        Parameters:
            mask: Mask of radar image Numpy array WxH
            radar_type (str): Radar type (77_ghz or 24_ghz) 
            ext (str): Extension to add to filename when saving
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world    
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        path = self._get_path(radar_type=radar_type, is_processed=True, x_angle=x_angle, y_angle=y_angle, z_angle=z_angle)
        masks_path = f'{path}/radar_masks{ext}.pkl'
        os.makedirs(os.path.dirname(masks_path), exist_ok=True)
        data = {
            'radar_mask': mask
        }
        with open(masks_path, "wb") as f:
            pickle.dump(data, f)

    def load_radar_masks(self, radar_type='77_ghz', ext='', angles=None):
        """
        Load mask for 2D radar image

        Parameters:
            radar_type (str): Radar type (77_ghz or 24_ghz) 
            ext (str): Extension to add to filename when saving
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world    
        Returns:
            Dictionary of data from radar mask
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        path = self._get_path(radar_type=radar_type, is_processed=True, x_angle=x_angle, y_angle=y_angle, z_angle=z_angle)
        masks_path = f'{path}/radar_masks{ext}.pkl'
        if not os.path.exists(masks_path):
            print(f'Couldnt find object: {masks_path}')
            return None
        with open(f'{masks_path}', 'rb') as f:
            masks = pickle.load(f)
        return masks

    def save_surface_normals(self, radar_type, image, x_locs, y_locs, z_locs, angles=None, antenna_locs=None, ext= "", channels=None, normals=None):
        """
        Saves estimated surface normals to correct directory
        Parameters:
            radar_type (str): the type of radar to use ("10_ghz", "24_ghz", "77_ghz")
            image (numpy array): processed SAR image by ImageProcessor
            x_locs (numpy array): sampled x locations
            y_locs (numpy array): sampled y locations
            z_locs (numpy array): sampled z locations
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world
            antenna_locs (numpy array): antenna locations used to processed image
            ext (str): extension of saved file name
            channels (numpy array): Radar channels
            normals (numpy array): Estimated normal vectors
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + f"/normals/normal_data{ext}.npz"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez_compressed(path, sum_img = image,
                       x_locs=x_locs, y_locs=y_locs, z_locs=z_locs, 
                       antenna_locs=np.array(antenna_locs),
                       channels=channels,
                       normals=normals)

    def load_surface_normals(self, radar_type, angles=None, ext= ""):
        """
        Loads estimated surface normals from correct directory
        Parameters:
            radar_type (str): the type of radar to use ("10_ghz", "24_ghz", "77_ghz")
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world
            ext (str): extension of saved file name

        Returns:
            sum_img (numpy array): processed SAR image by ImageProcessor
            x_locs (numpy array): sampled x locations
            y_locs (numpy array): sampled y locations
            z_locs (numpy array): sampled z locations
            antenna_locs (numpy array): antenna locations used to processed image
            channels (numpy array): Radar channels
            normals (numpy array): Estimated normal vectors
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + f"/normals/normal_data{ext}.npz"
        data = np.load(path,allow_pickle=True )
        sum_img=data['sum_img']
        x_locs=data['x_locs'] 
        y_locs=data['y_locs'] 
        z_locs=data['z_locs'] 
        antenna_locs=data['antenna_locs']
        channels=data['channels']
        normals=data['normals']
        return sum_img, (x_locs, y_locs, z_locs), antenna_locs, channels, normals
        
    def save_sdf_file(self, data, radar_type, angles=None, ext= ""):
        """
        Saves SDF and other data
        Parameters:
            data (dict): Dictionary of data
            radar_type (str): the type of radar to use ("10_ghz", "24_ghz", "77_ghz")
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world
            ext (str): extension of saved file name
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + f"/sdf/sdf{ext}.npz"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **data)

    def load_sdf_file(self, radar_type, angles=None, ext= ""):
        """
        Loads SDF and other data
        Parameters:
            radar_type (str): the type of radar to use ("10_ghz", "24_ghz", "77_ghz")
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world
            ext (str): extension of saved file name
            data (dict): Dictionary of data
        Returns:
            data (dict): Dictionary of SDF and other data
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + f"/sdf/sdf{ext}.npz"
        if not os.path.exists(path):
            print(f'Couldnt find object: {path}')
            return None
        data = np.load(path,allow_pickle=True )
        return data
    
    def save_optimization_file(self, data, radar_type, angles=None, ext= ""):
        """
        Saves output of optimization process
        Parameters:
            data (dict): Dictionary of data
            radar_type (str): the type of radar to use ("10_ghz", "24_ghz", "77_ghz")
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world
            ext (str): extension of saved file name
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + f"/optimization/opt_output{ext}.npz"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **data)

    def load_optimization_file(self, radar_type, angles=None, ext= "", load_old=False):
        """
        Loads output of optimization process
        Parameters:
            radar_type (str): the type of radar to use ("10_ghz", "24_ghz", "77_ghz")
            angles (tuple or None): Tuple of 3 numbers for x,y, and z angles, or None to automatically fill in the angles. 
                    If None, the angles will be fixed to 0,0,0 in simulation or determined based on the exp num in the real-world
            ext (str): extension of saved file name
            data (dict): Dictionary of data
        Returns:
            data (dict): Dictionary of SDF and other data
        """
        x_angle, y_angle, z_angle = self._parse_angles(angles)
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=True) + f"/optimization/opt_output{ext}.npz"
        if not os.path.exists(path):
            print(f'Couldnt find object: {path}')
            return None
        data = np.load(path,allow_pickle=True )
        return data
    

    def load_radar_files(self, radar_type, aperture_type='normal'):
        """
        Load raw radar files for processing (Note: this function is only needed when processing raw radar data into an image)
        
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns: 
            Loaded complex-valued radar (adc) files for the specified radar type
        """
        x_angle, y_angle, z_angle = self._find_obj_angles()
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False) + "/radar_data"
        all_data = {}
        filenames = sorted(os.listdir(path))
        params_dict = utilities.get_radar_parameters(radar_type=radar_type, is_sim=False, aperture_type=aperture_type)
        NUM_FRAMES = params_dict['num_frames']
        SAMPLES_PER_CHIRP = params_dict['num_samples']
        NUM_CHIRP = params_dict['num_chirps']
        for i, filename in enumerate(filenames):
            if radar_type == "24_ghz":

                # 24 GHz needs formatting before loading
                if filename[-4:] != '.txt': continue
                timestamp = int(filename[9:19])
                try:
                    adcData = np.loadtxt(f'{path}/{filename}', dtype=None, delimiter=',')
                except:
                    continue

                adcDataFormatted = np.zeros(((SAMPLES_PER_CHIRP * NUM_CHIRP * NUM_FRAMES), 2), dtype=np.complex64)
                if adcData.shape[0] == NUM_CHIRP * NUM_FRAMES*4:
                    for i in range(NUM_CHIRP * NUM_FRAMES):
                        adcDataFormatted[SAMPLES_PER_CHIRP * i: SAMPLES_PER_CHIRP * (i + 1), 0] = adcData[i * 4] + 1j * adcData[i * 4 + 1]
                        adcDataFormatted[SAMPLES_PER_CHIRP * i: SAMPLES_PER_CHIRP * (i + 1), 1] = adcData[i * 4 + 2] + 1j * adcData[i * 4 + 3]
                all_data[timestamp] = adcDataFormatted

            elif radar_type == "77_ghz":

                if filename[-4:] != '.bin': continue
                try:
                    timestamp = int(filename[8:18])
                except:
                    timestamp = int(filename[9:21])

                fid = open(f'{path}/{filename}', 'rb')
                adcData = np.fromfile(fid, dtype='<i2')
                numLanes = 4
                adcData = np.reshape(adcData, (int(adcData.shape[0] / (numLanes * 2)), numLanes * 2))
                adcData = adcData[:, [0, 1, 2, 3]] + 1j * adcData[:, [4, 5, 6, 7]]
                all_data[timestamp] = adcData

        return all_data

    def load_robot_loc_files(self, radar_type):
        """
        Load raw robot location files for processing (Note: this function is only needed when processing raw radar data into an image)
        
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns:
            Loaded continuous robot location (antenna) numpy (npy) files
        """
        x_angle, y_angle, z_angle = self._find_obj_angles()
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False)
        robot_loc_files = []
        for filename in os.listdir(path):
            if filename[:7] == 'antenna':
                robot_loc_files.append(filename)
        combined = {}
        robot_loc_files = sorted(robot_loc_files)
        for filename in robot_loc_files:
            cur_file = dict(np.load(f'{path}/{filename}'))
            if cur_file['times_77'].shape[0] != cur_file['tx_77_locs'].shape[0]:    
                for key in cur_file:
                    min_val = min(cur_file['times_77'].shape[0], cur_file['tx_77_locs'].shape[0])
                    cur_file['times_77'] = cur_file['times_77'][:min_val-20]
                    cur_file['tx_77_locs'] = cur_file['tx_77_locs'][:min_val-20]
            if cur_file['times_24'].shape[0] != cur_file['tx_24_locs'].shape[0]:    
                for key in cur_file:
                    min_val = min(cur_file['times_24'].shape[0], cur_file['tx_24_locs'].shape[0])
                    cur_file['times_24'] = cur_file['times_24'][:min_val-20]
                    cur_file['tx_24_locs'] = cur_file['tx_24_locs'][:min_val-20]
                    
            for key in cur_file:
                try:
                    combined[key] = np.concatenate((combined[key], cur_file[key]))
                except KeyError:
                    combined[key] = cur_file[key]
        return combined

    def load_robot_file(self, radar_type):
        """
        Load raw robot files for processing (Note: this function is only needed when processing raw radar data into an image)
        
        Loads robot pickle file labeled "simple" or "continuous." Simple robot data are 
        collected as the robot repeatedly moves and stops to collect data. Continuous robot
        data are collected as the robot continuously moves on each row.

        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns:
            Dictionary of robot data
        """
        x_angle, y_angle, z_angle = self._find_obj_angles()
        path = self._get_path(radar_type, x_angle, y_angle, z_angle, is_processed=False)
        robot_files = []
        for filename in os.listdir(path):
            if filename[:10] == 'continuous':
                robot_files.append(filename)
        robot_files = sorted(robot_files)
        robot_data = {}
        for filename in robot_files:
            with open(f'{path}/{filename}', 'rb') as f:
                cur_file = pickle.load(f)
                for key in cur_file:
                    if key == 'pattern':
                        continue
                    try:
                        robot_data[key] = robot_data[key] + cur_file[key]
                    except KeyError:
                        robot_data[key] = cur_file[key]
                
        robot_data['pattern'] = 0.5
        return robot_data
    
    def load_exp_json(self, radar_type):
        """ 
        Load the experiment json file.

        Parameters: 
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            x/y/z angle (int): rotation angle in degrees of the loading object
        Returns: 
            json data as a dictionary
        """
        x_angle, y_angle, z_angle = self._find_obj_angles()
        path_to_json = self._get_path(radar_type, x_angle, y_angle, z_angle, is_json=True)

        # Open and read the JSON file
        with open(f'{path_to_json}/experiment_info.json', 'r') as file:
            data = json.load(file)
        return data

    def load_all_data(self, radar_type):
        """
        Loads all data: robot location, radar data, and continuous robot location data (Note: this function is only needed when processing raw radar data into an image)
        
    
        Parameters:
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            angle (int): rotation angle in degrees of the loading object

        Returns:
            Dictionaries of robot data, radar data, and continuous robot location data
        """
        x_angle, y_angle, z_angle = self._find_obj_angles()
        exp_data = self.load_exp_json(radar_type)
        aperture_type = exp_data['aperture size']
        robot_data = self.load_robot_file(radar_type)
        radar_data = self.load_radar_files(radar_type, aperture_type=aperture_type)
        robot_loc_data = self.load_robot_loc_files(radar_type)
        return robot_data, radar_data, robot_loc_data, exp_data
