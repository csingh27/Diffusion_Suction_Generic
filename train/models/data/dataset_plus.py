'''
Author: HDT
'''

import os
import sys
from joblib import Parallel, delayed
import numpy as np
import torch
import h5py
import open3d as o3d
import torch.utils.data as data
from torchvision import transforms

FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
sys.path.append(FILE_DIR)

from pointcloud_transforms import PointCloudShuffle, ToTensor
from torch.utils.data import DataLoader



def collect_cycle_obj_sence_dir(data_dir, cycle_range, scene_range):
    dirs = []
    for cycle_id in range(cycle_range[0],cycle_range[1]):
        for scene_id in range(scene_range[0],scene_range[1]):
            dirs.append(os.path.join(data_dir, 'cycle_{:0>4}'.format(cycle_id),  '{:0>3}.h5'.format(scene_id)))      
    return dirs
   

def load_dataset_by_cycle_layer(dir, mode='train', collect_names=False):
    num_point_in_h5 = 16384

    f = h5py.File(dir)
    points = f['points'][:].reshape(num_point_in_h5, 3)*1000
    suction_or = f['suction_or'][:].reshape(num_point_in_h5, 3)
    suction_seal_scores = f['suction_seal_scores'][:]
    suction_wrench_scores = f['suction_wrench_scores'][:]
    suction_feasibility_scores = f['suction_feasibility_scores'][:]
    individual_object_size_lable = f['individual_object_size_lable'][:]

    dataset = {
        'points': points,
        'suction_or':suction_or,
        'suction_seal_scores': suction_seal_scores,
        'suction_wrench_scores': suction_wrench_scores,
        'suction_feasibility_scores': suction_feasibility_scores,
        'individual_object_size_lable': individual_object_size_lable, 
        }

    return dataset




class DiffusionSuctionNetDataset(data.Dataset):
    def __init__(self, data_dir,cycle_range, scene_range, mode='train', transforms=None,collect_names=False):
        self.mode = mode
        self.collect_names = collect_names
        self.transforms = transforms
        self.dataset_dir = collect_cycle_obj_sence_dir(data_dir, cycle_range, scene_range)

    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, idx):
        # ----------------------------------------------------------加载h5文件的数据
        dataset = load_dataset_by_cycle_layer(self.dataset_dir[idx])


        sample = {
            'points': dataset['points'].copy().astype(np.float32),
            'suction_or': dataset['suction_or'].copy().astype(np.float32),
            'suction_seal_scores': dataset['suction_seal_scores'].copy().astype(np.float32),
            'suction_wrench_scores': dataset['suction_wrench_scores'].copy().astype(np.float32),
            'suction_feasibility_scores': dataset['suction_feasibility_scores'].copy().astype(np.float32),
            'individual_object_size_lable': dataset['individual_object_size_lable'].copy().astype(np.float32),
        }

        # # ----------------------------------------------------------收集加载数据的信息
        if self.collect_names:
            cycle_temp = self.dataset_dir[idx].split('/')[-2]
            cycle_index = int(cycle_temp.split('_')[1])
            obj_and_scene_temp = self.dataset_dir[idx].split('/')[-1]
            scene_index = int(obj_and_scene_temp[0:3])
            name = [cycle_index,scene_index]
            sample['name'] = name

        # ---------------------------------------------------------数据进行转化
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


