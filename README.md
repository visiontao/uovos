# Unsupervised Online Video Object Segmentation with Motion Property Understanding
Tao Zhuo, Zhiyong Cheng*, Peng Zhang*, Yongkang Wong, and Mohan Kankanhalli

# Results on DAVIS-2016 TrainVal Dataset (50 videos)

 | Measure   |  NLC  |  LMP   | FSEG  | ARP  | UOVOS | 
 |-----------|-------|--------|-------|------|-------|
 | J Mean    |  64.1 |  69.7  |  71.6 | 76.3 | 77.8  |
 | J Recall  |  73.1 |  82.9  |  87.7 | 89.2 | 93.6  | 
 | J Decay   |  8.6  |  5.6   |  1.7  | 3.6  | 2.1   | 
 | F Mean    |  59.3 |  66.3  |  65.8 | 71.1 | 72.0  |
 | F mean    |  65.8 |  78.3  |  79.0 | 82.8 | 87.7  | 
 | F mean    |  8.6  |  6.7   |  4.3  | 7.3  | 3.8   |
 | T         |  36.6 |  68.8  |  29.5 | 35.9 | 33.0  |

NLC: Video Segmentation by Non-Local Consensus voting. A. Faktor, M. Irani, BMVC 2014. \
LMP: Learning Motion Patterns in Videos. P. Tokmakov, K. Alahari, C. Schmid, CVPR 2017. \
FSEG: FusionSeg: Learning to combine motion and appearance for fully automatic segmentation of generic objects in videos. S. Jain, B. Xiong, K. Grauman, CVPR 2017. \
ARP: Primary Object Segmentation in Videos Based on Region Augmentation and Reduction. Y.J. Koh, C.-S. Kim, CVPR 2017. 

# Setup
Ubuntu \
Matlab \
Python2.7 \
Mask-RCNN https://github.com/matterport/Mask_RCNN \ 
Opencv_3.4

# Citation:
If you use this code, please cite the following paper:

@article{zhuo2018unsupervised,
  title={Unsupervised Online Video Object Segmentation with Motion Property Understanding},
  author={Zhuo, Tao and Cheng, Zhiyong and Zhang, Peng and Wong, Yongkang and Kankanhalli, Mohan},
  journal={arXiv preprint arXiv:1810.03783},
  year={2018}
}

# Contact
Tao Zhuo (zhuotao@nus.edu.sg)

