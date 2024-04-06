# LiDAR2Map
> Song Wang, Wentong Li, Wenyu Liu, Xiaolu Liu, Jianke Zhu*

This is the official implementation of **LiDAR2Map: In Defense of LiDAR-Based Semantic Map Construction Using Online Camera Distillation** (CVPR 2023)  [[Paper](https://arxiv.org/pdf/2304.11379.pdf)] [[Video](https://youtu.be/nr25xFZbx8U?si=P8n6bl0-9Cx3uq2u)].

<p align="center"> <a><img src="fig/framework.png" width="90%"></a> </p>

## Preparation
### nuScene download
Please download the whole nuScene dataset from the [official website](https://www.nuscenes.org/nuscenes).

### Environment setup
Our project is built with [Pytorch](https://pytorch.org/get-started/locally/) >= 1.7 and revised [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) from [BEVFusion](https://github.com/mit-han-lab/bevfusion).

You can install the [tree-filter](https://github.com/megvii-research/TreeEnergyLoss) by:
```bash
cd ./map/model/loss/kernels/lib_tree_filter
python3 setup.py build develop
```


## Training and Inference
To train the model from scratch, you can run:
```bash
cd ./map
bash train.sh # multi-gpu
python train_lidar2map.py # single-gpu
```
To inference with the obtained checkpoint, you can run:
```bash
python test.py --modelf /path/to/ckpt # single-gpu
```

## Acknowledgements

Thanks for the pioneer work in online map learning:
[HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet)
[BEVFusion](https://github.com/mit-han-lab/bevfusion)
[BEVerse](https://github.com/zhangyp15/BEVerse)



## Citations
```
@inproceedings{wang2023lidar2map,
      title={LiDAR2Map: In Defense of LiDAR-Based Semantic Map Construction Using Online Camera Distillation},
      author={Wang, Song and Li, Wentong and Liu, Wenyu and Liu, Xiaolu and Zhu, Jianke},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={5186--5195},
      year={2023}
}
```
