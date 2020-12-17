# thesisWork


## How to run with sample data

1. Download target dataset from Animal-10:https://www.kaggle.com/alessiocorrado99/animals10
2. Only use the five classes: dog, horse, butterfly, cat, sheep
3. modify the source_data and target_data path within DeepCoral or DAN code
4. the source_data can be any data_path provided in 'data/' folder



## How to form new source dataset
1. Downloading all the data from ImageNet, description in 'data' folder
2. Use /util/selector.py to select the 'positive' sister term images
3. if you only want to use identical label images in source domain, you can just randomly select it
4. use the selected sister term images and identical label images in source domain. Make the folder structure same as the sample provided 


## Overview of the whole training process
1. refer to the target label, collect identical label images from https://github.com/mf1024/ImageNet-Datasets-Downloader
2. randomly select these identical label images into source domain. For example, 30 images for each classes
3. Refer to WordNet sister term of each target label, collect these sister term images from https://github.com/mf1024/ImageNet-Datasets-Downloader
4. Use selector.py to selecet the positive image by Resnet50 and top softmax score. Then add the images into source domain
5. modify the source and target data path in the code, you are able to train the model





# Reference

DAN and DeepCoral codes are both come from https://github.com/jindongwang/transferlearning/tree/master/code/deep
