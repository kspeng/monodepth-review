# Breif
This is a review note of monodepth method designed by author [mrharicot](https://github.com/mrharicot). The preliminary step is to reproduce the results using the source code by [mrharicot Github](https://github.com/mrharicot/monodepth.git). There are few unclear issues in author's instruction and they are clarified in this note. The next step is to improve the computational performance of this method. The note aims to provide a light neural net architecture to achieve real-time depth estimation (60fps) using the comercial gaming GPU (ex. Nvidia GTX 1050 Ti). 

# Preliminary 
There are two parts of the preproduction process - local host and hpc server (UofA HPC). It is infeasible to train the model in the local host, which may only process the single estimation, testing, and evaluation. However, all four processes will be applied in hpc server. 	

Before the main four processes, the pretrained models and the dataset, including Kitti, and citispaces, need to be ready. The pretrained models are available if you follow author's instruction. Datasets also can be downloaded but need to be reorganized and reformated to match author's configuration. The first step is to get the models and dataset ready. The following section will discuss the preprocessing of the dataset to match author's requirements.

## Dataset Preprocessing
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
Author used two different split of the data, **kitti** and **eigen**, amounting for respectively 29000 and 22600 training samples, we can find them in the source code - [filenames](utils/filenames) folder. These are for training purpose. For evaluation, we need [KITTI stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), providing 200 official training set pairs, and [Eigen NIPS14](http://www.cs.nyu.edu/~deigen/depth/), corresponding to the 697 test images.

KITTI dataset has all PNG format images (In total weight acorund **200GB**), which are all converted into JPG format in author's configuration. However, the depth images must be kept in PNG format. Otherwise, the evaluation process won't work. Simply say, only raw images are converted to JPG format.  

Next, in author's sample command, it implies that the training and testing folders of KITTI_stereo_2015 need to be allocated in data/kitti/ folder. 

### [Cityscapes](https://www.cityscapes-dataset.com)
We need to register in order to download the data, which already has a train/val/test/trainextra set with over 23000 training images.  Author used `leftImg8bit_trainvaltest.zip`, `rightImg8bit_trainvaltest.zip`, `leftImg8bit_trainextra.zip` and `rightImg8bit_trainextra.zip` which weights **110GB**.

Same as KITTI, author converted all raw images to JPG file. Author also put trainextra dataset into train dataset folder (data/citispaces/train). We have to do the same thing to ensure all raw images to be in the right pace. 

Then, we are ready to go for these fun processes.

## Local Host

Using cityspaces dataset and model as example.

1. Just want to try a single image
  ```shell
  python monodepth_simple.py --image_path data/test/test_img.jpg \
  --checkpoint_path models/cityspaces/model_cityscapes
  ```
2. Train our own model
  ```shell
  python monodepth_main.py --mode train --model_name my_model \
  --data_path data/citispaces/test/ --filenames_file \
  utils/filenames/cityscapes_test_files.txt --log_directory \
  log/ --batch_size 2 --num_epochs 1
  ```
3. Test pretrained model then evaluate the results
  - Test KITTI Stereo 2015
  ```shell
  python monodepth_main.py --mode test --data_path data/kitti/ \
  --filenames_file utils/filenames/kitti_stereo_2015_test_files.txt \
  --log_directory log/ --checkpoint_path models/kitti/model_kitti
  ```
  - Evaluate the results
  ```shell
  python utils/evaluate_kitti.py --split kitti --predicted_disp_path \
  models/kitti/disparities.npy --gt_path data/kitti/
  ```
  
## UA hpc server

Using cityspaces dataset and model as example. 
- The dataset are placed in \xdisk\data\.
- Assume we have the singularity container ready, refering to [Singularity Container for Computer Vision Python Tensorflow-GPU](https://github.com/kspeng/UA_HPC_Configuration/blob/master/Singularity%20Container%20for%20Computer%20Vision%20Python%20Tensorflow-GPU.md). The container .img file is in the path **~/workspace/envImg/**

1. Just want to try a single image
  ```shell
  singularity run --nv ~/workspace/envImg/keras+tensorflow_gpu-1.4.1-cp35-cuda8-cudnn6.img \
  monodepth_simple.py --image_path data/test/roadImage00.jpg --checkpoint_path \
  models/cityspaces/model_cityscapes.data-00000-of-00001
  ```
2. Train our own model
  ```shell
  singularity run --nv ~/workspace/envImg/keras+tensorflow_gpu-1.4.1-cp35-cuda8-cudnn6.img \
  monodepth_main.py --mode train --model_name my_model_kitti --data_path \
  ~/../../../xdisk/kspeng/data/kitti/ --filenames_file utils/filenames/kitti_train_files.txt \
  --log_directory ./log/ --batch_size 8 --num_epochs 50
  ```
3. Test pretrained model then evaluate the results
  - Test KITTI Stereo 2015
  ```shell
  singularity run --nv ~/workspace/envImg/keras+tensorflow_gpu-1.4.1-cp35-cuda8-cudnn6.img \
  monodepth_main.py --mode test --data_path ~/../../../xdisk/kspeng/data/kitti/ --filenames_file \
  utils/filenames/kitti_stereo_2015_test_files.txt --log_directory ~/tmp/ \
  --checkpoint_path models/kitti/model_kitti
  ```
  - Evaluate the results
  ```shell
  singularity run --nv ~/workspace/envImg/keras+tensorflow_gpu-1.4.1-cp35-cuda8-cudnn6.img \
  utils/evaluate_kitti.py --split kitti --predicted_disp_path models/kitti/disparities.npy \
  --gt_path ~/../../../xdisk/kspeng/data/kitti/
  ```

