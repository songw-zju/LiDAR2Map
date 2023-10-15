# LiDAR2Map
> Song Wang, Wentong Li, Wenyu Liu, Xiaolu Liu, Jianke Zhu*

This is the official implementation of **LiDAR2Map: In Defense of LiDAR-Based Semantic Map Construction Using Online Camera Distillation** (CVPR 2023)  [[Paper](https://arxiv.org/pdf/2304.11379.pdf)] [[Video](https://youtu.be/nr25xFZbx8U?si=P8n6bl0-9Cx3uq2u)].



## Abstract
Semantic map construction under bird's-eye view (BEV) plays an essential role in autonomous driving. In contrast to camera image, LiDAR provides the accurate 3D observations to project the captured 3D features onto BEV space inherently. However, the vanilla LiDAR-based BEV feature often contains many indefinite noises, where the spatial features have little texture and semantic cues. In this paper, we propose an effective LiDAR-based method to build semantic map. Specifically, we introduce a BEV pyramid feature decoder that learns the robust multi-scale BEV features for semantic map construction, which greatly boosts the accuracy of the LiDAR-based method. To mitigate the defects caused by lacking semantic cues in LiDAR data, we present an online Camera-to-LiDAR distillation scheme to facilitate the semantic learning from image to point cloud. Our distillation scheme consists of feature-level and logit-level distillation to absorb the semantic information from camera in BEV. The experimental results on challenging nuScenes dataset demonstrate the efficacy of our proposed LiDAR2Map on semantic map construction, which significantly outperforms the previous LiDAR-based methods over 27.9% mIoU and even performs better than the state-of-the-art camera-based approaches.



 ## TODO

- [ ] Add more instructions about the environment preparation

- [ ] Release pre-trained models of LiDAR2Map

  

## Framework
<p align="center"> <a><img src="fig/framework.png" width="90%"></a> </p>



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
