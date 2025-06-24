
import sys
import os

sys.path.append('..')
from utils.object_information import ObjectInformation, ObjectAttributes, ExperimentAttributes
from utils import utilities
from utils import generic_loader 
from utils import generic_loader 
from utils import visualization
import time
from image_processor import *

import argparse


parser = argparse.ArgumentParser(description="A script that process robot collected data to create mmWave image")
parser.add_argument("--name", type=str, default="" , help="Object Name. Only one of name or ID is required")
parser.add_argument("--id", type=str, default="", help="Object ID #. Only one of name or ID is required")
parser.add_argument("--ext", type=str, default="", help="Extension to save with")
parser.add_argument("--is_los", type=str, default="y", help="Is LOS? (y/n)")
parser.add_argument("--overwrite_existing", type=str, default="n", help="Overwrite existing file of the same name? If no and a file already exists, it will skip this step.")

args = parser.parse_args()
obj_name = args.name
obj_id = args.id
ext = args.ext
overwrite_existing = (args.overwrite_existing == 'y' or args.overwrite_existing == 'Y')
radar_type = "77_ghz" # We only use the 77_ghz radar for mmNorm
exp_num = "2" # We only use Exp 2 of the MITO dataset
is_los = (args.is_los == 'y' or args.is_los == 'Y')

assert radar_type in ['77_ghz', '24_ghz'], "Please choose a valid radar type."

# Fill in missing info
obj_info = ObjectInformation()
assert not (obj_name == '' and obj_id == ''), "Both name and ID can't be empty"
obj_id, obj_name = obj_info.fill_in_identifier_sep(obj_name, obj_id)

t1 = time.time()
# Set up loader and visualizer
loader = generic_loader.GenericLoader(obj_id, obj_name, is_sim=False, is_los=is_los, exp_num=exp_num)
processor = ImageProcessor(False)

# Check for an existing file, if applicable
skip = False
if not overwrite_existing: 
    # Try to load the file. This checks if it exists and is a valid file
    try:
        sum_img, (x_locs, y_locs, z_locs), antenna_locs, channels, normals = loader.load_surface_normals(radar_type=radar_type, ext=args.ext)
    except:
        sum_img = None
    if sum_img is not None:
        print(f'The output file already exists and this step was called with the overwrite_existing flag. Skipping this step.')
        skip = True

# Run the computation if applicable
if not skip:
    print(f'Running {radar_type} robot imager for {obj_id}_{obj_name} (LOS: {is_los}) ')

    # Processes the image
    robot_data, radar_data, robot_loc_data, exp_data = loader.load_all_data(radar_type)

    # Correlate location and measurements
    data = processor.correlate_locs_and_meas(radar_type, robot_data, radar_data, exp_data, robot_loc_data, speed="speed_8")

    # Generate sar image
    res = processor.generate_mmwave_surface_normals(radar_type, data, exp_data)
    sum_image, locs, normals, robot_locs, meas = res
    loader.save_surface_normals(radar_type, sum_image, *locs, angles=None, ext=args.ext, antenna_locs=robot_locs, channels=meas, normals=normals)
    t2 = time.time()
    print(f'Took {(t2-t1)} sec to finish processing image')

