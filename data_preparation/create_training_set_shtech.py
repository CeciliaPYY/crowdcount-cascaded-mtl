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


with open("../json/new.json", "r") as f:
    train_data = json.load(f)

num_val = math.ceil(num_images*0.1)

for i in range(num_images):
    if (i%10 == 0):
        print('Processing %3d/%d files\n', i, num_images)

    input_img_name = all_image_names[i]
    print(input_img_name)
    im = cv2.imread(input_img_name)
    h, w, c = im.shape
    if c == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    wn2 = w/8
    hn2 = h/8
    wn2 =8 * math.floor(wn2/8)
    hn2 =8 * math.floor(hn2/8)

    annPoints = train_data[os.path.split(input_img_name)[-1]]
    annPoints = np.array(annPoints)

    if( w <= 2*wn2 ):
        im = cv2.imresize(im,[ h,2*wn2+1])
        annPoints[:,0] = annPoints[:,0]*2*wn2/w
    if( h <= 2*hn2):
        im = cv2.imresize(im,[2*hn2+1,w])
        annPoints[:,1]= annPoints[:,1]*2*hn2/h

    h, w = im.shape
    a_w = wn2+1
    b_w = w - wn2
    a_h = hn2+1
    b_h = h - hn2

    im_density = get_density_map_gaussian(im,annPoints)

    for j in range(1,N+1):
        x = math.floor((b_w - a_w) * np.random.rand(1) + a_w)
        y = math.floor((b_h - a_h) * np.random.rand(1) + a_h)
        x1 = x - wn2
        y1 = y - hn2
        x2 = x + wn2-1
        y2 = y + hn2-1

        im_sampled = im[int(y1):int(y2),int(x1):int(x2)]
        im_density_sampled = im_density[int(y1):int(y2),int(x1):int(x2)]
        annPoints_sampled = annPoints[np.logical_and(np.logical_and(annPoints[:,0]<x2,
  annPoints[:,0]>x1),
                                                     np.logical_and(annPoints[:,1]<y2,
  annPoints[:,1]>y1))]

        img_idx = "{}_{}".format(((os.path.split(input_img_name)[-1]).split("_")[-1]).split(".")[0],j)

        if(i < num_val):
            cv2.imwrite(val_path_img+'{}.png'.format(img_idx), im_sampled)
            np.savetxt(val_path_den+'{}.csv'.format(img_idx), im_density_sampled, fmt='%10.5f', delimiter=",")
        else:
            cv2.imwrite(train_path_img+'{}.png'.format(img_idx), im_sampled)
            np.savetxt(train_path_den+'{}.csv'.format(img_idx), im_density_sampled, fmt='%10.5f', delimiter=",")
