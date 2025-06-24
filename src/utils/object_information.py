"""
This file is used to automatically load/find information about objects within the dataset
"""

import pandas as pd
from enum import Enum
import sys
sys.path.append('..')
from utils import utilities

# Different attributes defined for each object
class ObjectAttributes(str, Enum):
    NAME = 'Object'
    ID = 'ID Number'
    SIZE = 'Size (#)'
    SIZE_CLASS = 'Size (desc)'
    CATEGORY = 'Category'
    MATERIAL = 'Material'
    OFFSET_X = 'Angle offset x'
    OFFSET_Y = 'Angle offset y'
    OFFSET_Z = 'Angle offset z'

# Different attributes that change for each experiment
class ExperimentAttributes(str, Enum):
    NAME = 'Object'
    ID = 'ID Number'
    EXP_NUM = 'Experiment Number'
    CROP = 'Crop'
    CROP_HIGH = 'Crop High'
    SIM_AVAIL = 'Sim availability'
    LOS_AVAIL = 'LOS availability'
    NLOS_AVAIL = 'NLOS availability'
    LARGE_AP = 'Large Aperture'
    CALIBRATION_ID_24 = '24 Calibration ID'
    LOS_BACKGROUND_ID = 'Background ID LOS'
    NLOS_BACKGROUND_ID = 'Background ID NLOS'

class ObjectInformation():
    def __init__(self):
        """
        Initialize the object information. 
        Load two CSV files: one which contains information on the objects, and one which contains information on the individual experiments. 
        """
        self.object_df = pd.read_csv(f'{utilities.get_root_path()}/objects.csv')
        self.experiment_df = pd.read_csv(f'{utilities.get_root_path()}/experiments.csv', dtype={'24 Calibration ID': str, 'Background ID LOS': str, 'Background ID NLOS': str})
        self.object_df.dropna(how='all', inplace=True)
        self.object_df.dropna(axis=1, how='all', inplace=True)

    def fill_in_identifier_sep(self, obj_name='None', obj_id='None'):
        """
        Find the object name from the object ID, or vice versa.
        For this function, either obj_name, or obj_id can be None, but not both
        This function accepts both 'None' and None

        Parameters:
            obj_name: 'None', None, or a string of the object name
            obj_id: 'None', None, or a string of the object ID
        Returns: 
            obj_id: string of object id
            obj_name: string of object name

        """
        # Ensure both aren't empty
        assert not (obj_name == 'None' and obj_id == 'None'), "Both name and ID can't be empty"

        # Fill in obj ID if needed
        if obj_name == 'None' or obj_name is None:
            obj_name = self._get_object_name(obj_id)

        # Fill in obj name if needed
        elif obj_id == 'None' or obj_id is None:
            obj_id = self._get_object_id(obj_name)
        return obj_id, obj_name

    def fill_in_identifier(self, obj_name_or_id):
        """
        This function takes a single string of either an object name or ID. Then, it returns both the name and ID of the object

        Parameters:
            obj_name_or_id: String of either object name or object ID
        Returns:
            obj_id: string of object id
            obj_name: string of object name
        """
        # This string is an ID if any of the characters are a digit
        is_id = any(char.isdigit() for char in obj_name_or_id)
        if is_id:
            return self.fill_in_identifier_sep(obj_id=obj_name_or_id, obj_name='None')
        else:
            return self.fill_in_identifier_sep(obj_name=obj_name_or_id, obj_id='None')

    def get_object_info(self, attr, name=None, obj_id=None, exp_num=None):
        """
        Get information about an object or experiment

        Parameters:
            attr (ObjectAttributes or ExperimentAttributes): Attribute to read
            name (str): Name of object
            id (str): ID of object
            exp_num (str or None): Experiment number. This is required for ExperimentAttributes, and not necessary for ObjectAttributes
        Returns: 
            Requested value from given object/experiment
        """
        if type(attr) == ObjectAttributes:
            exp_num = None # Object attributes don't require an exp_num
            df = self.object_df
        else:
            if exp_num is None: raise Exception('Cannot access experiment-level information without an experiment number')
            df = self.experiment_df
            exp_num = int(exp_num)
        return self._get_one_info_from_object(col=attr.value, df=df, name=name, obj_id=obj_id, exp_num=exp_num)

    def get_object_offset(self, name=None, obj_id=None):
        """
        Get the X/Y/Z offset of an object as a list. 
        This is a helper function which calls get_object_info 3 times to access all 3 parameters
        
        Parameters:
            name (str): Name of object
            id (str): ID of object
        Returns:
            List of 3 values: x, y, and z offsets
        """
        return [self.get_object_info(ObjectAttributes.OFFSET_X, name=name, obj_id=obj_id),
                self.get_object_info(ObjectAttributes.OFFSET_Y, name=name, obj_id=obj_id),
                self.get_object_info(ObjectAttributes.OFFSET_Z, name=name, obj_id=obj_id)]

    def convert_partial_obj_list(self, objs):
        """
        Convert a list of object IDs to names, or vice versa

        Parameters: objs: list of object IDs or Names (as strings)
        Returns: List of tuples of (ids, names)
        """
        if objs[0][0] == '0':
            objs = self._convert_ids_to_names(objs)
        else:
            objs = self._convert_names_to_ids(objs)
        return objs

    def list_all_objects(self):
        """
        Returns: List of all IDs/Names in dataset
        """
        all_names = self.object_df["Object"]
        all_ids = [self._get_object_id(name) for name in all_names]
        return zip(all_ids, all_names)

    # # PRIVATE FUNCTIONS: 
    def _get_object_all_info(self, df, name=None, obj_id=None, exp_num=None):
        """
        Args: 
            df (dataframe): Dataframe object to search
            name (str): desired object name. Can be None if obj_id is not None
            obj_id (int): desired object id. Can be None if name is not None
            exp_num (str): Desired experiment number. If None, don't filter by exp number
        Returns:
            list of all object info (name, obj_id, size, category, etc.)
        """
        assert not (name is None and obj_id is None), "Either Name or ID needs to not be None"
        # Find all rows matching either name or ID
        if name is not None:
            info = df[df["Object"]==name]
        elif obj_id is not None:
            obj_id = obj_id.split('0')[1:] # Remove leading 0s
            if len(obj_id) == 2: 
                if len(obj_id[0]) == 0: obj_id = str(obj_id[1]) # Properly account for IDs with 2 leading 0s
                else: obj_id = str(obj_id[0] + '0') # Properly account for IDs ending in 0
            else: obj_id = str(obj_id[0])
            info = df[df["ID Number"]==str(obj_id)]

        # Apply experiment number filter if applicable
        if exp_num is not None:
            info = info[info["Experiment Number"]==exp_num]
        
        # Check if no results found, otherwise return all info
        if info.empty:
            print(f"No such object named {name} ID: {obj_id}")
            return None
        return info
        
    def _get_one_info_from_object(self, col, df, name=None, obj_id=None, exp_num=None):
        """
        Get a single value from a given object

        Parameters:
            col: Name of column in CSV file to read
            name (str): Name of object
            obj_id (str): ID of object
            exp_num (str): Desired experiment number. If None, don't filter by exp number
        """
        object_info = self._get_object_all_info(df, name=name, obj_id=obj_id,exp_num=exp_num)
        if object_info is not None:
            col_info = object_info[col].values[0]
            return col_info
        else:
            print(f'No {col} found for object, {name}')
            return None
        
    def _get_object_id(self, name):
        """
        Get Object ID given a name
        Parameters:
            name (str): Name of object
        Returns:
            String of ID of object
        """
        id_num = self._get_one_info_from_object("ID Number", self.object_df, name=name, obj_id=None)
        # Fill with missing zeros
        num_digits = sum(c.isdigit() for c in id_num)
        id_num = '0'*(3-num_digits) + id_num
        return id_num
    
    def _get_object_name(self, obj_id):
        """
        Get Object name given an ID
        Parameters:
            obj_id (str): ID of object
        Returns:
            name (str): Name of object
        """
        return self._get_one_info_from_object("Object", self.object_df, name=None, obj_id=obj_id)

    def _convert_names_to_ids(self, obj_names):
        """
        Convert a list of names to a list of object IDs+Names
        Parameters:
            obj_names (list): List of object names
        Returns: List of tuples of (IDs, names)
        """
        objs = []
        for name in obj_names:
            obj_id = self._get_object_id(name=name)
            objs.append((obj_id, name))
        return objs
    
    def _convert_ids_to_names(self, obj_ids):
        """
        Convert a list of IDs to a list of object IDs+Names
        Parameters:
            obj_ids(list): List of object IDs
        Returns: List of tuples of (IDs, names)
        """
        objs = []
        for obj_id in obj_ids:
            name = self._get_object_name(obj_id=obj_id)
            objs.append((obj_id, name))
        return objs
