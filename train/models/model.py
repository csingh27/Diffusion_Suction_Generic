""" 
Pytorch version of dsnet.
Author: HDT
"""
import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)


import torch
import torch.nn as nn
import numpy as np
import backbone2
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from typing import Union, Dict, Tuple, Optional


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // ratio, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x




class ScheduledCNNRefine(nn.Module):
    def __init__(self, channels_in = 128, channels_noise = 4, **kwargs):
        super().__init__(**kwargs)
        self.noise_embedding = nn.Sequential(
            nn.Conv1d(channels_noise, 64, 1),
            nn.GroupNorm(4, 64),
            # 不能用batch norm，会统计输入方差，方差会不停的变
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(4, 128),
            nn.ReLU(True),
            nn.Conv1d(128, channels_in, 1),
        )

        self.time_embedding = nn.Embedding(1280, channels_in)


        self.pred = nn.Sequential(
            nn.Conv1d(channels_in, 64, 1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(4, 128),
            nn.ReLU(True),
            nn.Conv1d(128, channels_noise, 1),
        )

        self.channelattention = ChannelAttention(128)
        self.spatialattention = SpatialAttention()


    def forward(self, noisy_image, t, feat):
        if t.numel() == 1:
            feat = feat + self.time_embedding(t)[..., None] # feat( n ,16384,128   )   time_embedding(t) (128) 
        else:
            feat = feat + self.time_embedding(t)[..., None,]
        feat = feat + self.noise_embedding(noisy_image.permute(0, 2, 1))

        feat = self.channelattention(feat)
        feat = self.spatialattention(feat)

        ret = self.pred(feat)+noisy_image.permute(0, 2, 1)

        return ret


class CNNDDIMPipiline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            features,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            **kwargs,
    ) -> Union[Dict, Tuple]:
        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            raise RuntimeError(
                "generator.device == 'cpu'",
                "0.11.0",
                message,
            )
            generator = None

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # timesteps 选择了20步
            # 1. predict noise model_output
            model_output = self.model(image, t.to(device), features)

            model_output = model_output.permute(0, 2, 1)

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample']
            # np.savez("score_"+str(t)+'.npz', image.cpu().numpy())

        return image

class dsnet(nn.Module):
    def __init__(self, use_vis_branch, return_loss):
        super().__init__()
        self.use_vis_branch = use_vis_branch
        self.loss_weights =  {
                                'suction_seal_scores_head': 50.0, 
                                'suction_wrench_scores_head': 50.0,
                                'suction_feasibility_scores_head': 50.0,
                                'individual_object_size_lable_head': 50.0,
                                }
        self.return_loss = return_loss
        
        backbone_config = {
            'npoint_per_layer': [4096,1024,256,64],
            'radius_per_layer': [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]],
            'input_feature_dims':3,
        }
        self.backbone = backbone2.Pointnet2MSGBackbone(**backbone_config)
        backbone_feature_dim = 128
        
        # self.suction_seal_scores_head = self._build_head([backbone_feature_dim, 64, 64, 1])
        # self.suction_wrench_scores_head = self._build_head([backbone_feature_dim, 64, 64, 1])
        # self.suction_feasibility_scores_head = self._build_head([backbone_feature_dim, 64, 64, 1])
        # self.individual_object_size_lable_head = self._build_head([backbone_feature_dim, 64, 64, 1])
    


        # add diffusion
        self.model = ScheduledCNNRefine(channels_in=backbone_feature_dim, channels_noise=4 )
        self.diffusion_inference_steps = 20
        num_train_timesteps=1000
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)

        self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)
        self.bit_scale = 0.5

        


    def ddim_loss(self, condit, gt,):
        # Sample noise to add to the images
        noise = torch.randn(gt.shape).to(gt.device)
        bs = gt.shape[0]

        gt_norm = (gt - 0.5) * 2 * self.bit_scale

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt.device).long()
        # 这里的随机是在 bs维度，这个情况不能太小。
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_norm, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, condit)
        noise_pred = noise_pred.permute(0, 2, 1)

        loss = F.mse_loss(noise_pred, noise)

        return loss




    def forward(self, inputs):
        ###inputs['point_clouds']: batch_size*num_point*3
        batch_size = inputs['point_clouds'].shape[0]
        num_point = inputs['point_clouds'].shape[1]
        
        # -----------------------------------------------------pointnet++提取堆叠场景点云
        input_points = inputs['point_clouds']  # torch.Size([4, 16384, 3])
        input_points = torch.cat((input_points, inputs['labels']['suction_or']), dim=2)
        features, global_features = self.backbone(input_points)


        
        
        if self.return_loss:  # calculate pose loss
            s1 = inputs['labels']['suction_seal_scores'].unsqueeze(-1)
            s2 = inputs['labels']['suction_wrench_scores'].unsqueeze(-1)
            s3 = inputs['labels']['suction_feasibility_scores'].unsqueeze(-1)
            s4 = inputs['labels']['individual_object_size_lable'].unsqueeze(-1)
            gt = torch.cat((s1, s2, s3, s4), dim=2)
            
            pred_results = self.pipeline(   
                batch_size=batch_size,
                device=features.device,
                dtype=features.dtype,
                shape=(16384,4),
                features = features,
                num_inference_steps=self.diffusion_inference_steps,
            )

            # self-diff
            # ddim_loss1 = self.ddim_loss(features, pred_results)
            ddim_loss1 = self.ddim_loss(features, gt)


            ddim_loss2 = F.mse_loss(pred_results, gt)
            ddim_loss = [ddim_loss1, ddim_loss2]
            
            pred_results = None
        else:
            pred_results = self.pipeline(   
                batch_size=batch_size,
                device=features.device,
                dtype=features.dtype,
                shape=(16384,4),
                features = features,
                num_inference_steps=self.diffusion_inference_steps,
            )
            ddim_loss = None
        return pred_results, ddim_loss


    


    def visibility_loss(self, pred_vis, vis_label):
        loss = torch.mean( torch.abs(pred_vis - vis_label) )
        return loss


    def _compute_loss(self, preds_flatten, labels):

        batch_size, num_point = labels['suction_seal_scores'].shape[0:2]
        suction_seal_scores_label_flatten = labels['suction_seal_scores'].view(batch_size * num_point)  # (B*N,)
        suction_wrench_scores_flatten = labels['suction_wrench_scores'].view(batch_size * num_point)  # (B*N,)
        suction_feasibility_scores_label_flatten = labels['suction_feasibility_scores'].view(batch_size * num_point)  # (B*N,)
        individual_object_size_lable_flatten = labels['individual_object_size_lable'].view(batch_size * num_point)  # (B*N,)
        
        pred_suction_seal_scores, pred_suction_wrench_scores, pred_suction_feasibility_scores,pred_individual_object_size_lable = preds_flatten
        
        losses = dict()
        losses['suction_seal_scores_head'] = self.visibility_loss(pred_suction_seal_scores, suction_seal_scores_label_flatten) * self.loss_weights['suction_seal_scores_head'] 
        losses['suction_wrench_scores_head'] = self.visibility_loss(pred_suction_wrench_scores, suction_wrench_scores_flatten) * self.loss_weights['suction_wrench_scores_head'] 
        losses['suction_feasibility_scores_head'] = self.visibility_loss(pred_suction_feasibility_scores, suction_feasibility_scores_label_flatten) * self.loss_weights['suction_feasibility_scores_head'] 
        losses['individual_object_size_lable_head'] = self.visibility_loss(pred_individual_object_size_lable, individual_object_size_lable_flatten) * self.loss_weights['individual_object_size_lable_head'] 
        losses['total'] = losses['suction_seal_scores_head'] + losses['suction_wrench_scores_head'] + losses['suction_feasibility_scores_head'] + losses['individual_object_size_lable_head'] 
        
        return losses

    def _build_head(self, nchannels):
        assert len(nchannels) > 1
        num_layers = len(nchannels) - 1

        head = nn.Sequential()
        for idx in range(num_layers):
            if idx != num_layers - 1:
                head.add_module( "conv_%d"%(idx+1), nn.Conv1d(nchannels[idx], nchannels[idx+1], 1))
                head.add_module( "bn_%d"%(idx+1), nn.BatchNorm1d(nchannels[idx+1]))
                head.add_module( "relu_%d"%(idx+1), nn.ReLU())
            else:   # last layer don't have bn and relu
                head.add_module( "conv_%d"%(idx+1), nn.Conv1d(nchannels[idx], nchannels[idx+1], 1))
        return head




# Helper function for saving and loading network
def load_checkpoint(checkpoint_path, net, map_location=None,optimizer=None):
    """ 
    Load checkpoint for network and optimizer.
    Args:
        checkpoint_path: str
        net: torch.nn.Module
        optimizer(optional): torch.optim.Optimizer or None
    Returns:
        net: torch.nn.Module
        optimizer: torch.optim.Optimizer
        start_epoch: int
    """
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda: 3'))
    checkpoint = torch.load(checkpoint_path,map_location=map_location)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(checkpoint_path, start_epoch))
    return net, optimizer, start_epoch

def save_checkpoint(checkpoint_path, current_epoch, net, optimizer, loss):
    """ 
    Save checkpoint for network and optimizer.
    Args:
        checkpoint_path: str
        current_epoch: int, current epoch index
        net: torch.nn.Module
        optimizer: torch.optim.Optimizer or None
        loss:
    """
    save_dict = {'epoch': current_epoch+1, # after training one epoch, the start_epoch should be epoch+1
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }
    try: # with nn.DataParallel() the net is added as a submodule of DataParallel
        save_dict['model_state_dict'] = net.module.state_dict()
    except:
        save_dict['model_state_dict'] = net.state_dict()
    torch.save(save_dict, checkpoint_path)

def save_pth(pth_path, current_epoch, net, optimizer, loss):
    save_dict = {'epoch': current_epoch+1, # after training one epoch, the start_epoch should be epoch+1
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }
    try: # with nn.DataParallel() the net is added as a submodule of DataParallel
        save_dict['model_state_dict'] = net.module.state_dict()
    except:
        save_dict['model_state_dict'] = net.state_dict()
    torch.save(save_dict, pth_path + '.pth')


