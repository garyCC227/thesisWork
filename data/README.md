## Description

Here are some sample data that are used in source domain during adapatation. These data are used for 10/20/30 identical label images case.
 - 10ID == I used 10 identical label images in source domain for adapatation for each class.
 - 10sis == Used 10 'positive' sister term images in source domain for adapatation for each class.
   'positive' images means: applying my data selection method to filter 'useful' images from sister term images
 
 
## Data source
 - all of these data are collected from ImageNet via https://github.com/mf1024/ImageNet-Datasets-Downloader
 - the images I selected are according to WordNet Princeton search
 
 
 
## How to select identical label images to use in source domain
 - collecte identical label in ImageNet via data source link
 - randomly select 10/20/30 images for each classes
 
 
## How to select 'positive' sister term images to use in source domain
 - discover the sister term of target label via WordNet
 - filter the 'positive' sister term images by my selection method
 - randomly select 10/20 'positive' sister images add into source domain
