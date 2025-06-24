"""" This file contains code for visualizing results. 
You can run this file to visualize a single image, or import it to access the visualization functions.  """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import copy
import sys
import open3d as o3d

sys.path.append('..')
from utils import utilities
from utils.object_information import ObjectInformation, ObjectAttributes
from utils.generic_loader import *

class Visualizer:
    """
    Visualizer includes helper functions to visualize 3D images or 1D range profiles
    """

    def plot_sar_image(self, image, x_locs, y_locs, z_locs, plot_dim=2, normalization=None, plot_all_slices=False, obj_name='', title=None):
        """
        Plot Synthetic Aperture Radar (SAR) image 

        Parameters:
            image (numpy array): image to plot. Size: (L,W,H)
            x_locs (np array): x coordinates for each voxel in the image. Size: (L,)
            y_locs (np array): y coordinates for each voxel in the image. Size: (W,)
            z_locs (np array): z coordinates for each voxel in the image. Size: (H,)
            plot_dim (bool): the axis to project the data over. 0 = x, 1 = y, 2 = z (default: 2). 
                        E.g., plot_dim=2 would be an overhead view while 0 or 1 are side views
            normalization (matlabe Normalize): Normalization to apply for plotting or None to use default
            plot_all_slices (bool): Plot all individual image slices before plotting average image (along plot dim). Default: False
            obj_name (str): Object name to include in title of plot or an empty string to ignore
        Returns: None
        """
        # Compute magnitude of image
        abs_image = copy.deepcopy(np.abs(image))

        title_start = 'SAR Image' if obj_name == '' else f'SAR Image of {obj_name}'
        if plot_dim == 0: iter_locs = x_locs; axis_1_locs = y_locs; axis_2_locs = z_locs; axis_1_label='Y'; axis_2_label = 'Z'
        if plot_dim == 1: iter_locs = y_locs; axis_1_locs = x_locs; axis_2_locs = z_locs; axis_1_label='X'; axis_2_label = 'Z'
        if plot_dim == 2: iter_locs = z_locs; axis_1_locs = x_locs; axis_2_locs = y_locs; axis_1_label='X'; axis_2_label = 'Y'

        # Plot individual image slices
        if plot_all_slices:
            # Create normalization if not provided
            if normalization is None:
                normalization = Normalize(vmin=0, vmax=np.max(abs_image[:,:], axis=None))

            # Plot all slices
            for iter_loc in range(len(iter_locs)):
                if plot_dim == 0: image_slice = abs_image[iter_loc, :, :]; title=f'{title_start}: X = {x_locs[iter_loc]}'
                if plot_dim == 1: image_slice = abs_image[:, iter_loc, :]; title=f'{title_start}: Y = {y_locs[iter_loc]}'
                if plot_dim == 2: image_slice = abs_image[:, :, iter_loc]; title=f'{title_start}: Z = {z_locs[iter_loc]}'
                
                plt.pcolormesh(axis_1_locs, axis_2_locs, image_slice.T, norm=normalization, cmap = 'jet')
                plt.colorbar()
                plt.title(title)
                plt.xlabel(axis_1_label)
                plt.ylabel(axis_2_label)
                plt.axis('equal')
                plt.show()

        # Average the image along the plotting dimension
        avg_image = np.sum(abs_image[:,:,:], axis=plot_dim) / abs_image.shape[plot_dim]
        # avg_image = avg_image**2

        # Create normalization if not provided
        if normalization is None:
            normalization = Normalize(vmin=0, vmax=np.max(avg_image, axis=None)*0.5)

        # Plot average image
        if title is None:
            if plot_dim == 0: title=f'{title_start}: Avg along X'
            if plot_dim == 1: title=f'{title_start}: Avg along Y'
            if plot_dim == 2: title=f'{title_start}: Avg along Z'
        plt.pcolormesh(axis_1_locs, axis_2_locs, avg_image.T, norm=normalization, cmap = 'jet',linewidth=0,rasterized=True)
        plt.colorbar()
        plt.title(title)
        plt.xlabel(axis_1_label)
        plt.ylabel(axis_2_label)
        plt.axis('equal')
        # title = int(int(title.split('_')[-1])/4)
        # plt.savefig(f'/home/ldodds/clip_ap2/{title}.png')
        plt.show()
        
    def visualize_reconstruction(self, obj_id, name, is_los, ext, plot_baseline=False):
        loader = GenericLoader(obj_id, name, is_sim=False, is_los=is_los, exp_num='2') # Note: Only exp 2 is supported at this time
        data = loader.load_optimization_file(radar_type='77_ghz', ext=f'_optimization_result{ext}')
        points = data['points']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd])



if __name__=='__main__':
    """ This will plot a single reconstruction """
    # Experiment to plot
    obj_name = "spoon" # Name of object to plot
    obj_id = "031" # ID of the object (or none to fill automatically) 
    is_los = False # Whether to plot LOS or NLOS

    # Get object ID
    if obj_id is None:
        obj_info = ObjectInformation()
        obj_id, obj_name = obj_info.fill_in_identifier_sep(obj_name, obj_id)

    # Visualize reconstruction
    visualizer = Visualizer()
    visualizer.visualize_reconstruction(obj_id, obj_name, is_los, ext='')


