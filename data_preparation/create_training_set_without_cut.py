import numpy as np
import os
from random import randint
import cv2
import json
import math
import random
from src.get_density_map_gaussian_xuan import get_density_map_gaussian

seed = 95461354
np.random.seed(seed)
N = 9
dataset = 'B'
dataset_name = 'shanghaitech_part_{}_patches_{}'.format(dataset, N)
path = '../data-peng/original/shanghaitech/part_{}_final/train_data/images/'.format(dataset)
output_path = '../data-peng/formatted_trainval/'
train_path_img = output_path + dataset_name + '/train/'
train_path_den = output_path + dataset_name + '/train_den/'
val_path_img = output_path + dataset_name + '/val/'
val_path_den = output_path + dataset_name + '/val_den/'

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(train_path_img):
    os.mkdir(train_path_img)
if not os.path.exists(train_path_den):
    os.mkdir(train_path_den)
if not os.path.exists(val_path_img):
    os.mkdir(val_path_img)
if not os.path.exists(val_path_den):
    os.mkdir(val_path_den)

all_image_names = os.listdir(path)
all_image_names = [os.path.join(path,i) for i in all_image_names]
num_images = len(all_image_names)


with open("../data-peng/json/train_new.json", "r") as f:
    train_data = json.load(f)

num_val = math.ceil(num_images*0.1)

for i in range(num_images):
    if (i%10 == 0):
        print('Processing {}/{} files\n'.format(i, num_images))

    input_img_name = all_image_names[i]
    print(input_img_name)
    im = cv2.imread(input_img_name)
    h, w, c = im.shape
    if c == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    annPoints = train_data[os.path.split(input_img_name)[-1]]
    annPoints = np.array(annPoints)
    im_density = get_density_map_gaussian(im,annPoints)

    if(i < num_val):
        cv2.imwrite(val_path_img+'{}.png'.format(os.path.basename(input_img_name).split(".")[0]), im)
        np.savetxt(val_path_den+'{}.csv'.format(os.path.basename(input_img_name).split(".")[0]), im_density, fmt='%10.5f', delimiter=",")
    else:
        cv2.imwrite(train_path_img+'{}.png'.format(os.path.basename(input_img_name).split(".")[0]), im)
        np.savetxt(train_path_den+'{}.csv'.format(os.path.basename(input_img_name).split(".")[0]), im_density, fmt='%10.5f', delimiter=",")
