import os
from itertools import islice
import numpy as np

# input function
def read_kitty(dir_names:list, return_labels:bool=True, max_len:int=None, input_folder:str='input'):
    """
    Opens a point cloud dataset following kitty formatting
    
    Parameters:
    dir_names: list of names of directorys containing a kitty sequence
    lables: if true, reads and returns labels as well
    max_len: maximum amount of point clouds to read from each directory
    input_folder: name of head folder containing the datasets
    
    Returns: Dictionaries for scans and optionally labels. Each with directory
    name as first key, file name as second key and list as value
    """

    scans = {n:dict() for n in dir_names}
    for dir_name in dir_names:
        # get datapaths
        scan_path = os.path.join(input_folder, dir_name, 'velodyne')
        scan_names = sorted([os.path.join(dp, f) 
                      for dp, dn, fn in os.walk(os.path.expanduser(scan_path)) 
                      for f in fn])
        
        # get filenames
        scan_path_len = len(scan_path) + len('/')
        file_names = [scan[scan_path_len:-4] for scan in scan_names]
        
        # open and read pointcloud
        for file_name in islice(file_names, max_len):
            with open(scan_path+'/'+file_name+'.bin', mode='rb') as file:
                scan = np.fromfile(file, dtype=np.float32)
                scan = np.reshape(scan, (-1,4))
                scans[dir_name].update({file_name:scan})
           
    if not return_labels: # break early
        return scans
 
    labels = {n:dict() for n in dir_names}
    for dir_name in dir_names:
        # get datapaths
        label_path = os.path.join(input_folder, dir_name, 'labels')
        label_names = sorted([os.path.join(dp, f) 
                   for dp, dn, fn in os.walk(os.path.expanduser(label_path)) 
                   for f in fn])
        
        # get filenames
        label_path_len = len(label_path) + len('/')
        file_names = [scan[label_path_len:-6] for scan in label_names]
        
        # open and read labels 
        for file_name in islice(file_names, max_len):
            with open(label_path+'/'+file_name+'.label', mode='rb') as file:
                label = np.fromfile(file, dtype=np.int16)
                label = np.reshape(label, (-1,2))
                labels[dir_name].update({file_name:label})
    
        # assertions
        matching_file_names = all(a == b for a,b in zip(scans[dir_name].keys(),labels[dir_name].keys()))
        assert matching_file_names, f'file name missmatch in dir {dir_name}'
        one_to_one_ratio = all(a.shape[0] == b.shape[0] for a,b in zip(scans[dir_name].values(),labels[dir_name].values()))
        assert one_to_one_ratio, f'data length missmatch in dir {dir_name}'
        
    return scans, labels 

def write_one_kitty(dir_name:str, scans, labels=None):
    "writes numpy point cloud data as kitty format"
    output_path = os.path.join(dir_name)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(dir_name, 'velodyne'))
        os.mkdir(os.path.join(dir_name, 'labels'))
        
    # make velodyne data
    vel_dir = os.path.join(dir_name, 'velodyne')
    for name, data in scans.items():
        data.tofile(f'{vel_dir}/{name}.bin', sep='', format='%s')
    
    # make label data
    if labels:
        lab_dir = os.path.join(dir_name, 'labels')
        for name, data in labels.items():
            data.tofile(f'{lab_dir}/{name}.label', sep='', format='%s') 

# output function
def write_kitty(scans:dict, labels:dict=None, folder:str='output'):
    """
    Writes a point cloud dataset following kitty formatting
    
    Parameters:
    scans: dictonary with directory name as first key, file name as second key and a list of point clouds as value
    labels: optional dictonary with directory name as first key, file name as second key and a list of labels as value
    folder: name of the top folder to put all output files and folders into
    """
    if not os.path.isdir(folder):
        os.mkdir(os.path.join(folder))

    for key in scans.keys():
        kwargs = {'labels':None} if not labels else {'labels':labels[key]}
        write_one_kitty(os.path.join(folder,key), scans[key], **kwargs)
