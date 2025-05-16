import numpy as np
import torch

class PointCloudShuffle(object):
    def __init__(self):
        self.num_point = 16384

    def __call__(self, sample):
        pt_idxs = np.arange(0, self.num_point)
        np.random.shuffle(pt_idxs)

        #将打乱之后的点云和标签对应上
        sample['points'] = sample['points'][pt_idxs]
        sample['suction_or'] = sample['suction_or'][pt_idxs]
        sample['suction_seal_scores'] = sample['suction_seal_scores'][pt_idxs]
        sample['suction_wrench_scores'] = sample['suction_wrench_scores'][pt_idxs]
        sample['suction_feasibility_scores'] = sample['suction_feasibility_scores'][pt_idxs]
        sample['individual_object_size_lable'] = sample['individual_object_size_lable'][pt_idxs]
        return sample

class ToTensor(object):
    def __call__(self, sample):
        sample['points'] = torch.from_numpy(sample['points'])
        sample['suction_or'] = torch.from_numpy(sample['suction_or'])
        sample['suction_seal_scores'] = torch.from_numpy(sample['suction_seal_scores'])
        sample['suction_wrench_scores'] = torch.from_numpy(sample['suction_wrench_scores'])
        sample['suction_feasibility_scores'] = torch.from_numpy(sample['suction_feasibility_scores'])
        sample['individual_object_size_lable'] = torch.from_numpy(sample['individual_object_size_lable'])
        return sample
