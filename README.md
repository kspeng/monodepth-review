# Breif
This is a review note of monodepth method designed by author mrharicot (https://github.com/mrharicot). The preliminary step is to reproduce the results using the source code by mrharicot (https://github.com/mrharicot/monodepth.git). There are few unclear issues in author's instruction and they are clarified in this note. The next step is to improve the computational performance of this method. The note aims to provide a light neural net architecture to achieve real-time depth estimation (60fps) using the comercial gaming GPU (ex. Nvidia GTX 1050 Ti). 

# Preliminary 
There are two parts of the preproduction process - local host and hpc server (UofA HPC). It is infeasible to train the model in the local host, which may only process the single estimation, testing, and evaluation. However, all four processes will be applied in hpc server. 	

Before the main four processes, the pretrained models and the dataset, including Kitti, and citispaces, need to be ready. The pretrained models are available if you follow author's instruction. Datasets also can be downloaded but need to be reorganized and reformated to match author's configuration. The first step is to get the models and dataset ready. The following section will discuss the preprocessing of the dataset to match author's requirements.

## Dataset Preprocessing
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
Author used two different split of the data, **kitti** and **eigen**, amounting for respectively 29000 and 22600 training samples, we can find them in the source code - [filenames](utils/filenames) folder. These are for training purpose. For evaluation, we need [KITTI stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), providing 200 official training set pairs, and [Eigen NIPS14](http://www.cs.nyu.edu/~deigen/depth/), corresponding to the 697 test images.

KITTI dataset has all PNG format images (In total acorund 200Gb), which are all converted into JPG format in author's configuration. However, the depth images must be kept in PNG format. Otherwise, the evaluation process won't work. Simply say, only raw images are converted to JPG format.  

Next, in author's sample command, it implies that the training and testing folders of KITTI_stereo_2015 need to be allocated in data/kitti/ folder. 

### [Cityscapes](https://www.cityscapes-dataset.com)
We need to register in order to download the data, which already has a train/val/test/trainextra set with over 23000 training images.  Author used `leftImg8bit_trainvaltest.zip`, `rightImg8bit_trainvaltest.zip`, `leftImg8bit_trainextra.zip` and `rightImg8bit_trainextra.zip` which weights **110GB**.

Same as KITTI, author converted all raw images to JPG file. Author also put trainextra dataset into train dataset folder (data/citispaces/train). We have to do the same thing to ensure all raw images to be in the right pace. 

Then, we are ready to go for these fun processes.

## Local Host

