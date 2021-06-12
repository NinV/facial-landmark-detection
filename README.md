Download weight for CNN backbone from this [link](https://drive.google.com/drive/folders/1sOXYoV_EIkdZRm0ROErQlmK-r-RG8m7o?usp=sharing)
Pretrained for HRNet on ImageNet from this [link](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
# Train data 300W and test COFW
* Download data from COFW [Link](https://drive.google.com/file/d/1bL0wl8lGTt3083qcaUwUXYT3EYX2g6Zj/view?usp=sharing')
* Download data from 300W [Link](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
* Run : python train_hrnet.py  --model backbone -i '300W' --annotation '300W' --test_images //test_annotations --test_annotation /COFW_test_color.mat --save_best_only --dataset 300W_COFW