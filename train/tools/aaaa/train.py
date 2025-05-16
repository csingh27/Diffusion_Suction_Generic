""" 
Author: dingtao huang
for diffusion_scution_net
"""


gpus_is = True # 多卡
gpus_is = False

# python -m torch.distributed.launch --nproc_per_node=2 train.py
if gpus_is:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,0'
else:
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
PROJECT_NAME = os.path.basename(FILE_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
sys.path.append(ROOT_DIR)

import math
from datetime import datetime
import h5py
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from models.model import dsnet, load_checkpoint, save_checkpoint
from models.utils.train_helper import BNMomentumScheduler, OptimizerLRScheduler, SimpleLogger
from models.data.pointcloud_transforms import PointCloudShuffle, ToTensor
from models.data.dataset_plus import DiffusionSuctionNetDataset



if gpus_is :
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='59882') 
        torch.distributed.init_process_group(backend="nccl",init_method=dist_init_method, world_size=world_size,rank=rank)

else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






# --------------------------------------------------------------------------------------------默认参数
BATCH_SIZE = 8
MAX_EPOCH = 500  #最大训练epoch
TRAIN_DATA_HOLD_EPOCH = 3  # 一个cycle训练多少个epoch
EVAL_STAP = 10 # 多少个epoch  val一次
DISPLAY_BATCH_STEP = 100  #  多少个batch数据进行打印loss
SAVE_STAP = 50 # 多少个epoch保存权重文件

# ------------------------------------------学习率和bn层参数
BASE_LEARNING_RATE = 0.001
LR_DECAY_RATE = 0.7
MIN_LR = 1e-6 
LR_DECAY_STEP = 80
LR_LAMBDA = lambda epoch: max(BASE_LEARNING_RATE * LR_DECAY_RATE**(int(epoch / LR_DECAY_STEP)), MIN_LR)
# bn decay
# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MIN = 0.001
BN_DECAY_STEP = LR_DECAY_STEP
BN_DECAY_RATE = 0.5
BN_LAMBDA = lambda epoch: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(epoch / BN_DECAY_STEP)), BN_MOMENTUM_MIN)
# --------------------------------------------------------------------------------------------默认参数


def generate_list(start, end, step):
    result = []
    for i in range(start, end, step):
        result.append([i,i+step])
    return result



# --------------------------------------------------------------------------------------------需要修改的参数
LOG_NAME = 'train'  #训练保存路径
# CHECKPOINT_PATH = 'checkpoint.tar'
CHECKPOINT_PATH = None
DATASET_DIR = '/opt/data/private/suctionnet-Packag/data_gen_for_package/h5_dataset_change/train'
TRAIN_CYCLE_RANGES = generate_list(0, 480, 20)                           
TRAIN_SCENE_RANGE = [1,51]  
TEST_CYCLE_RANGE =  [480, 490]
TEST_SCENE_RANGE = [1,51]  
# --------------------------------------------------------------------------------------------需要修改的参数







# --------------------------------------------------------------------------------------------SimpleLogger和tensorboard进行初始化
log_dir = os.path.join(ROOT_DIR, 'logs', PROJECT_NAME, LOG_NAME)
logger = SimpleLogger(log_dir, FILE_PATH)
SummaryWriter_log_dir = os.path.join(ROOT_DIR, 'logs', PROJECT_NAME, LOG_NAME, "tensorboard")
if not os.path.exists(SummaryWriter_log_dir): os.mkdir(SummaryWriter_log_dir)




# --------------------------------------------------------------------------------------------网络进行初始化
if gpus_is:
    # ( use_vis_branch, return_loss)
    net = dsnet(  True, True)
    net =  net.to(device)
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    num_gpus = torch.cuda.device_count()

    if CHECKPOINT_PATH is not None:
        net, optimizer, start_epoch = load_checkpoint(CHECKPOINT_PATH, net, None)
    else:
        start_epoch = 0

    if num_gpus > 1:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    if dist.get_rank() == 0:
        writer = SummaryWriter(SummaryWriter_log_dir)   
else:
    # ( use_vis_branch, return_loss)
    net = dsnet( True, True)
    net.to(device)
    writer = SummaryWriter(SummaryWriter_log_dir)

    if CHECKPOINT_PATH is not None:
        net, optimizer, start_epoch = load_checkpoint(CHECKPOINT_PATH, net, None)
    else:
        start_epoch = 0



# --------------------------------------------------------------------------------------------优化器 bn lr 进行初始化
optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=BN_LAMBDA, last_epoch=start_epoch-1)
lr_scheduler = OptimizerLRScheduler(optimizer, lr_lambda=LR_LAMBDA, last_epoch=start_epoch-1)


# --------------------------------------------------------------------------------------------transforms 进行初始化
transforms = transforms.Compose(
    [
        PointCloudShuffle(),# 把点乱序
        ToTensor()
    ]
)


# --------------------------------------------------------------------------------------------val数据集加载 进行初始化
if gpus_is:
    if dist.get_rank() == 0:
        print('Loading test dataset')
    test_dataset = DiffusionSuctionNetDataset(DATASET_DIR, TEST_CYCLE_RANGE, TEST_SCENE_RANGE, transforms=transforms)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=test_sampler)
    if dist.get_rank() == 0:
        print('Test dataset loaded, test point cloud size:', len(test_dataset.dataset_dir))
else:
    print('Loading test dataset')
    test_dataset = DiffusionSuctionNetDataset(DATASET_DIR, TEST_CYCLE_RANGE, TEST_SCENE_RANGE, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('Test dataset loaded, test point cloud size:', len(test_dataset.dataset_dir))
train_dataset = None





def train_one_epoch(loader,epoch):
    logger.reset_state_dict('train loss1','train loss2')
    if gpus_is:
        if dist.get_rank() == 0:
            logger.log_string('----------------TRAIN STATUS---------------')
    else:
            logger.log_string('----------------TRAIN STATUS---------------')
    net.train() 
    

    for batch_idx, batch_samples in enumerate(loader):
        if batch_idx == 2:
            start_time = time.time()
        
        # ---------------------------------------------------------------data 
        xyz_noise = torch.from_numpy(np.random.standard_normal(batch_samples['points'].shape)).float()
        # 检测数据  show_points([batch_samples['point_clouds'][0]])
        input_points_with_noise = batch_samples['points'] + xyz_noise * 2
        labels = {
            'suction_or': batch_samples['suction_or'].to(device),
            'suction_seal_scores': batch_samples['suction_seal_scores'].to(device),
            'suction_wrench_scores': batch_samples['suction_wrench_scores'].to(device),
            'suction_feasibility_scores': batch_samples['suction_feasibility_scores'].to(device),
            'individual_object_size_lable': batch_samples['individual_object_size_lable'].to(device),
        }
        inputs = {
            'point_clouds': input_points_with_noise.to(device),
            'labels': labels
        }
        # ---------------------------------------------------------------data 

        optimizer.zero_grad()
        _, losses = net(inputs)
        losses_all = losses[0] + losses[1]
        losses_all.backward()
        optimizer.step()
        log_state_dict = {
                            'train loss1': losses[0].item(),
                            'train loss2': losses[1].item(),
                            
                            }
        logger.update_state_dict(log_state_dict)

        if gpus_is:
            if dist.get_rank() == 0:
                if batch_idx % DISPLAY_BATCH_STEP == 0 and batch_idx!= 0:
                    print('Current batch/total batch num: %d/%d'%(batch_idx,len(loader)))
                    logger.print_state_dict(log=False)
                if batch_idx == 2:
                    t = time.time() - start_time
                    print('Successfully train one batchsize in %f seconds.' % (t))

        else:                                    
                if batch_idx % DISPLAY_BATCH_STEP == 0 and batch_idx!= 0:
                    print('Current batch/total batch num: %d/%d'%(batch_idx,len(loader)))
                    logger.print_state_dict(log=False)
                if batch_idx == 2:
                    t = time.time() - start_time
                    print('Successfully train one batchsize in %f seconds.' % (t))
                        #MAX_EPOCH*t* 1200 /BATCH_SIZE
                

    if gpus_is:
        if dist.get_rank() == 0:
            print('Current batch/total batch num: %d/%d'%(len(loader),len(loader)))
            logger.print_state_dict(log=True)
            loss_info = logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                writer.add_scalar(k, v, epoch)
    else:
            print('Current batch/total batch num: %d/%d'%(len(loader),len(loader)))
            logger.print_state_dict(log=True)
            loss_info = logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                writer.add_scalar(k, v, epoch)



def eval_one_epoch(loader,epoch):
    logger.reset_state_dict('eval loss1','eval loss2')
        
    if gpus_is:
        if dist.get_rank() == 0:
            logger.log_string('----------------EVAL STATUS---------------')
    else:
            logger.log_string('----------------EVAL STATUS---------------')


    net.eval() 
    loss_sum = 0
    for batch_idx, batch_samples in enumerate(loader):
        xyz_noise = torch.from_numpy(np.random.standard_normal(batch_samples['points'].shape)).float()
        input_points_with_noise = batch_samples['points'] + xyz_noise*2
        labels = {
            'suction_or': batch_samples['suction_or'].to(device),
            'suction_seal_scores': batch_samples['suction_seal_scores'].to(device),
            'suction_wrench_scores': batch_samples['suction_wrench_scores'].to(device),
            'suction_feasibility_scores': batch_samples['suction_feasibility_scores'].to(device),
            'individual_object_size_lable': batch_samples['individual_object_size_lable'].to(device),
        }
        inputs = {
            'point_clouds': input_points_with_noise.to(device),
            'labels': labels
        }

        with torch.no_grad():
            _, losses = net(inputs)
            losses_all = losses[0] + losses[1]
            loss_sum += losses_all.item()
            log_state_dict = {
                                'eval loss1': losses[0].item(),
                                'eval loss2': losses[1].item(), 
                                }
            logger.update_state_dict(log_state_dict)
                         
    if gpus_is:
        if dist.get_rank() == 0:
            logger.print_state_dict(log=True)
            loss_info = logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                writer.add_scalar(k, v, epoch)

    else:
            logger.print_state_dict(log=True)
            loss_info = logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                writer.add_scalar(k, v, epoch)         
           
    return loss_sum


def train(start_epoch):
    global train_dataset
    min_loss = 1e10
    for epoch in range(start_epoch, MAX_EPOCH): 
        # --------------------------------------------------------------------------------------------train数据集加载 
        if epoch%TRAIN_DATA_HOLD_EPOCH == 0 or train_dataset is None:
            cid = int(epoch/TRAIN_DATA_HOLD_EPOCH) % len(TRAIN_CYCLE_RANGES)
            if gpus_is:
                if dist.get_rank() == 0:
                    print('Loading train dataset...')
                train_dataset = DiffusionSuctionNetDataset(DATASET_DIR, TRAIN_CYCLE_RANGES[cid], TRAIN_SCENE_RANGE, transforms=transforms)
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=train_sampler)
                if dist.get_rank() == 0:
                    print('Test dataset loaded, test point cloud size:', len(train_dataset.dataset_dir))

            else:
                print('Loading train dataset...')
                train_dataset = DiffusionSuctionNetDataset(DATASET_DIR, TRAIN_CYCLE_RANGES[cid], TRAIN_SCENE_RANGE, transforms=transforms)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
                print('Test dataset loaded, test point cloud size:', len(train_dataset.dataset_dir))
        # --------------------------------------------------------------------------------------------train数据集加载 

        # --------------------------------------------------------------------------------------------配置lr 和 bn 参数
        bnm_scheduler.step(epoch) 
        lr_scheduler.step(epoch)
        if gpus_is:
            if dist.get_rank() == 0:
                logger.log_string('************** EPOCH %03d **************' % (epoch))
                logger.log_string(str(datetime.now()))
                logger.log_string('Current learning rate: %s'%str(lr_scheduler.get_optimizer_lr()))
                logger.log_string('Current BN decay momentum: %f'%(bnm_scheduler.get_bn_momentum(epoch)))
        else:
            logger.log_string('************** EPOCH %03d **************' % (epoch))
            logger.log_string(str(datetime.now()))
            logger.log_string('Current learning rate: %s'%str(lr_scheduler.get_optimizer_lr()))
            logger.log_string('Current BN decay momentum: %f'%(bnm_scheduler.get_bn_momentum(epoch)))
        train_one_epoch(train_loader,epoch)
        if epoch%EVAL_STAP == 0 and epoch>50:
            loss = eval_one_epoch(test_loader,epoch)
            if loss < min_loss:
                min_loss = loss
                save_checkpoint(os.path.join(log_dir, 'checkpoint.tar'), epoch, net, optimizer, loss)
                logger.log_string("Model saved in file: %s" % os.path.join(log_dir, 'checkpoint.tar'))
        if epoch%SAVE_STAP == 0 and epoch>50:
            save_checkpoint(os.path.join(log_dir, str(epoch)+'_'+'checkpoint.tar'), epoch, net, optimizer, loss)
    print(f'训练的场景完成！！！！！！！！！！！！！')


if __name__ == '__main__':
    train(start_epoch)
