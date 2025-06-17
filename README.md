## Diffusion Suction Grasping with Large-Scale Parcel Dataset


Illustration of the  suction-diffusion-denoising  process.
![Alt text](/images/1.gif)





This is the code of pytorch version for paper: [**Diffusion Suction Grasping with Large-Scale Parcel Dataset**]


## Overview of Diffusion-Suction architecture.
Illustration of the Diffusion-Suction architecture for 6DoF Pose Estimation in stacked scenarios.
![Alt text](/images/model1.png)

## Overview of Parcel-Suction-Dataset.
Illustration of the Self-Parcel-Suction-Labeling pipeline.
![Alt text](/images/model2.png)


## Qualitative results
Evaluation SuctionNet-1Billion dataset
![Alt text](/images/dataset1.png)
Evaluation Parcel-Suction-Dataset dataset
![Alt text](/images/dataset2.png)



## Getting Started

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/TAO-TAO-TAO-TAO-TAO/Diffusion_Suction.git
```
Install the environment：

Install [Pytorch](https://pytorch.org/get-started/locally/). It is required that you have access to GPUs. The code is tested with Ubuntu 16.04/18.04, CUDA 10.0 and cuDNN v7.4, python3.6.
Our backbone PointNet++ is borrowed from [pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch).
.Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd train\Sparepart\train.py
    python train.py install


Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'
    torch==1.1.0
    torchvision==0.3.0
    sklearn
    h5py
    nibabel


    

### 2. Train Diffusion-Suction
    cd train\Sparepart\train.py
    python train.py 



### 3. Evaluation on the custom data


Parcel-Suction-Dataset is available at [here](https://drive.google.com/drive/folders/1l4jz7LE7HXdn2evylodggReTTnip7J1Q?usp=sharing).


SuctionNet-1Billion is available at [here](https://github.com/graspnet/suctionnetAPI).


Evaluation metric
The python code of evaluation metric is available at [here](https://github.com/graspnet/suctionnetAPI).




## Citation
If you find our work useful in your research, please consider citing:

    @article{huang2025diffusion,
    title={Diffusion Suction Grasping with Large-Scale Parcel Dataset},
    author={Huang, Ding-Tao and He, Xinyi and Hua, Debei and Yu, Dongfang and Lin, En-Te and Zeng, Long},
    journal={arXiv preprint arXiv:2502.07238},
    year={2025}
    }

    @inproceedings{huang2024sd,
    title={Diffusion Suction Grasping with Large-Scale Parcel Dataset},
    author={dingtao huang, Debei Hua, Dongfang Yu, Xinyi He, Ente Lin, lianghong wang, Jinliang Hou, Long Zeng},
    booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2025},
    organization={IEEE}
    }





## Contact

If you have any questions, please feel free to contact the authors. 

Ding-Tao Huang: [hdt22@mails.tsinghua.edu.cn](hdt22@mails.tsinghua.edu.cn)

