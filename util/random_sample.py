import os
import shutil
import random

def copyfile_from_comb_list_one_animal(animal, comb_lists):
#   output_root = fr"./5species_10classes/comb/"
  for index, comb in enumerate(comb_lists):
    for species_path in comb:
      dir_path = species_path
      files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
#       print(dir_path,len(files))
      random_amount = 300 // len(comb)
      for x in range(random_amount):
          if len(files) == 0:
              break
          else:
              file = random.choice(files)
              outputdir = fr"./lvl2_final/15_15/comb{index}/{animal}" #TO MODIFY WHEN need
              if not os.path.isdir(outputdir):
                os.makedirs(outputdir)
              shutil.copy(file, outputdir)



animals = ['butterfly', 'cat', 'dog', 'horse', 'sheep']
selected_animals = ['dog']

##NOTE: Only do dog and horse, other just copy the whole folder
# selected_animals = ['dog', 'horse']
changed_animals = ['dog', 'horse', 'cat', 'sheep']
number_of_comb = 5 #TO MODIFY
lvl0_species = 15 #TODO
lvl1_species = 15 #TODO
for animal in animals:
  if animal not in selected_animals:
    continue
  
  #choose species from one sister level : 0
  lvl0_dir_ = fr"./whole2/{animal}" #TODO
  dirs = [os.path.join(lvl0_dir_,d) for d in os.listdir(lvl0_dir_)]
  lvl0_combs = []
  for index in range(number_of_comb):
    if len(dirs) >= lvl0_species:
      tmp = random.sample(dirs, lvl0_species)
      lvl0_combs.append(tmp)
    else:
      lvl0_combs.append(dirs)
 
  lvl1_dir_ = fr"./lvl2_horse_whole/{animal}" #TODO
  dirs = [os.path.join(lvl1_dir_,d) for d in os.listdir(lvl1_dir_)]
  
 
  #create comb for one animal
  lvl1_combs = []
  for index in range(number_of_comb):
    if len(dirs) >= lvl1_species:
      tmp = random.sample(dirs, lvl1_species)
      lvl1_combs.append(tmp)
    else:
      lvl1_combs.append(dirs)
  
  assert(len(lvl1_combs) == len(lvl0_combs))
  comb_lists = []
  for lvl0, lvl1 in zip(lvl0_combs, lvl1_combs):
    assert(len(lvl0) == lvl0_species)
    assert(len(lvl1) == lvl1_species)
    print(len(lvl0), len(lvl1), animal)
    tmp = lvl0 + lvl1
    comb_lists.append(tmp)
  
  copyfile_from_comb_list_one_animal(animal, comb_lists)

#rename
folder1 = "./lvl2_final/15_15" #TO modify
# folder2 = "./30species_5classes"

label_dict ={
'butterfly':'farfalla',
 'cat':'gatto',
 'chick':'gallina',
 'cow':'mucca',
 'dog':'cane',
 'elephant':'elefante',
 'horse':'cavallo',
 'sheep':'pecora',
 'spider':'ragno',
 'squirrel':'scoiattolo'
}
animals = ['butterfly', 'cat', 'dog', 'horse', 'sheep']

#rename -> e.g dog to cane. 
folder = folder1
for root in os.listdir(folder):
  comb_path = os.path.join(folder, root)
  for animal in os.listdir(comb_path):
    animal_path = os.path.join(comb_path, animal)
    if animal in animals:
      os.rename(animal_path,  os.path.join(comb_path, label_dict[animal]))


folder = folder1
for root in os.listdir(folder):
  comb_path = os.path.join(folder, root)
  for animal in os.listdir(comb_path):
    animal_path = os.path.join(comb_path, animal)
#     if animal in animals:
    files = [file for file in os.listdir(animal_path) if os.path.isfile(os.path.join(animal_path, file))]
    print(animal_path, len(files))