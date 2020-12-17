from pathlib import Path
import os
from PIL import *
import numpy as np
from keras.preprocessing import image
import random
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions



class Selector(object):
  """
  select positive sister term images by ResNet50
  """
  def __init__(self, label, data_path):
    self.animal = label
    self.data_path = data_path
    self.model = ResNet50(include_top=True, input_shape=(224,224,3), weights="imagenet", pooling="avg",)
  
  def create_data(self, files):
    data = []
    for i, img_path in enumerate(files):
        # print(img_paths)
        img = image.load_img(img_path, target_size=(224,224))
        img = image.img_to_array(img)
        data.append(img)

    data = np.array(data)
      
    return data

  def select(self):
    # animal = "cane"/"cavallo"/"farfalla"/"gatto"/"pecora"
    source_path = self.data_path

    #get source data
    dir_path = source_path
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    X = self.create_data(files)
    X = np.array(X)
    
    result = self.model.predict(X, verbose=1)
    second_softmax = np.argsort(result, axis=1)[:, -1] #Top softmax
    
    count = 0
    indexs = []
    for index, i in enumerate(second_softmax):
      if self.animal == 'cane':
        # dog range: 151 - 268: these are refered to WordNet label 
        if 151<=i and i <= 268:
          count += 1
          indexs.append(index)
      
      if self.animal == 'cavallo':
        # horse: 339, 603
        if i == 339 or i == 603:
          count += 1
          indexs.append(index)

      if self.animal == 'gatto':
        cat:281 - 287
        if 281<=i and i <= 287:
          count += 1
          indexs.append(index)
      
      if self.animal == 'farfalla':
        #butterfly:321 - 327
        if 321<=i and i <= 327:
          count += 1
          indexs.append(index)

      if self.animal == 'pecora':
        #sheep:348 - 353
        if 348<=i and i <= 353:
          count += 1
          indexs.append(index)
    
    F = np.array(files)
    indexs = np.array(indexs)
    filter_data = F[indexs]

    return filter_data


  def copyfile(self, animal, files, path):
    for index, file in enumerate(files):
      if len(files) == 0:
        break
      else:
        outputdir = fr"{path}/{animal}" #TO MODIFY WHEN need
        if not os.path.isdir(outputdir):
          os.makedirs(outputdir)
        shutil.copy(file, outputdir)











if __name__ == "__main__":
  path = rf'E:\code\data\lvl1_whole\butterfly\moth'
  animal = "farfalla"
  tmp = Selector(animal, path)
  
  print(tmp.select())