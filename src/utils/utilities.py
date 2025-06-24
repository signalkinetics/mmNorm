"""
This file contains various utility functions
"""
import json
import os
import argparse
import requests
import numpy as np
import matplotlib.pyplot as plt

def get_radar_parameters(radar_type='77_ghz', is_sim=False, aperture_type='normal'):
    """
    Load the radar parameters from the json file.

    Parameters:
        radar_type (str): '77_ghz' or '24_ghz'
        is_sim (str): Is simulation data
    Returns:
        radar parameters (dictionary)
    """
    if aperture_type == 'large': aperture_type = 'normal' # "Large" apertures have same radar parameters as a normal aperture
    params = load_param_json()
    current = params['simulation' if is_sim else 'robot_collected'][radar_type]
    if not is_sim:
        current = current[aperture_type]
    return current

def check_for_sam_weights():
    """
    This function checks if the SAM weights exists, and downloads them if they do not
    It will look for SAM weights in the path specified in the params.json

    Returns: The path to the weights
    """
    weight_path = get_sam_path()
    # Check for existing weights
    if os.path.isfile(weight_path): 
        print("SAM weights found")
    else: # Download weights if they do not exist
        desired_link = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" 
        print("SAM weights not found. Will start download from ",desired_link)
        if not os.path.exists(os.path.dirname(weight_path)):
            os.mkdir(os.path.dirname(weight_path))
        response = requests.get(desired_link)
        with open(weight_path, 'wb') as f:
            f.write(response.content)
    return weight_path
        

def get_sam_path():
    """
    This function constructs the SAM path based on the path in params.json. 
    If the provided path is relative, it will construct an absolute path

    Returns: An absolute path to where the weights should be
    Note: This function does not confirm whether the weights exists. Use check_for_sam_weights for that functionality
    """
    params = load_param_json()
    path = params['processing']['path_to_sam_weights']
    if path[0] == '/':
        # This is an absolute path, just return it
        return path 
    else:
        # This is a relative path, add the root directory
        return f'{get_root_path()}/{path}' 
    
def load_param_json():
    """
    Load the param.json file

    Returns: Nested dictionaries of parameters from json file
    """
    f = open(f'{get_root_path()}/src/utils/params.json')
    params = json.load(f)
    return params

def get_root_path():
    """
    Returns the path to the root of the repo
    """
    cwd = os.path.abspath(os.path.dirname(__file__))
    return f'{cwd}/../..'

def parse_tuple(tuple_str):
    """
    This function converts a string of comma separated list of integers to a tuple
    This function can be used in combination with argparse to accept a tuple of integers as a parameter
    When using this function, arguments can take the form of a comma separated list of integers (without parenthesis). For example: "1,2,3,4"

    Parameters: 
        tuple_str: Input string containing comma separated list of integers
    Returns: Tuple of integer elements
    """
    if tuple_str == 'None':
        return
    try:
        # Split the input string by comma and convert elements to integers
        elements = tuple(map(int, tuple_str.split(',')))
        return tuple(elements)
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple should be a comma-separated list of integers")


def convert_depth_to_xyz(z):
    """
    Converts a depth image to XYZ points
    Parameters:
        - z: A depth image (Shape: WxH)
    
    Returns: 
        - xyz: XYZ points corresponding to the depth image (Shape: WxHx3)
    """
    K = np.array([[618.1739501953125, 0.0, 321.27392578125], [0.0, 617.7586059570312, 241.05874633789062], [0.0, 0.0, 1.0]]) # Intrinsic matrix for realsense camera
    cam_fx = K[0,0]
    cam_fy = K[1,1]
    u0 = K[0,2]
    v0 = K[1,2]
    UMAX = z.shape[1]
    VMAX = z.shape[0]
    u_map = np.ones((VMAX, 1)) * np.arange(1, UMAX + 1) - u0  # u-u0
    v_map = np.arange(1, VMAX + 1).reshape(VMAX, 1) * np.ones((1, UMAX)) - v0  # v-v0
    x = u_map * z / cam_fx
    y = v_map * z / cam_fy

    # Convert from mm to m
    z = z.astype(x.dtype)
    x /=1000.0
    y /=1000.0
    z /=1000.0
    xyz = np.concatenate([x[:,:,np.newaxis],y[:,:,np.newaxis],z[:,:,np.newaxis]], axis=2)
    return xyz

def plot_dual_bar(x_vals, lower_val, middle_val, upper_val, lower_val2, middle_val2, upper_val2, xtick_labels=[], bar_labels=[], line_labels=[], xlabel='', ylabel='', ylabel2='', title=''):
    """
    Plotting function for dual bar plot
    """
    err_bars = np.zeros((2, len(lower_val)))
    err_bars[0, :] = np.array(middle_val) - np.array(lower_val)
    err_bars[1, :] = np.array(upper_val) - np.array(middle_val)

    if lower_val2 is not None:
        err_bars2 = np.zeros((2, len(lower_val2)))
        err_bars2[0, :] = np.array(middle_val2) - np.array(lower_val2)
        err_bars2[1, :] = np.array(upper_val2) - np.array(middle_val2)
    else:
        err_bars2 = None


    barWidth = 0.2 * (x_vals[1] - x_vals[0])
    # Set position of bar on X axis
    br1 = [x - barWidth/2 for x in x_vals]
    br2 = [x + barWidth/2 for x in x_vals]
    
    # Make the plot
    lines = []
    fig, ax = plt.subplots()
    line = ax.bar(br1, middle_val, yerr=err_bars,  width = barWidth, label=bar_labels[0], error_kw=dict(lw=5, capsize=5, capthick=3))
    lines.append(line)
    line = ax.bar(br2, middle_val2, yerr=err_bars2,  width = barWidth, label=bar_labels[1], error_kw=dict(lw=5, capsize=5, capthick=3))
    lines.append(line)
  
    ax.set_xticks(x_vals)   
    plt.xticks(x_vals, 
            xtick_labels)  
    plt.xlabel(xlabel)
    plt.title(title)
    legend = plt.legend(bbox_to_anchor=[0.175,0.75], loc='lower left')
    plt.axvline((x_vals[0]+x_vals[1])/2, linestyle='dashed', linewidth=4, color='black')
    legend.get_frame().set_alpha(None)
    fig = plt.gcf()
    fig.set_size_inches((13.5, 8.5), forward=False)
    plt.show()

