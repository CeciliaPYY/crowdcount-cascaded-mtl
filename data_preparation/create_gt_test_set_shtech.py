import numpy as np
import os
import cv2
from src.get_density_map_gaussian_xuan import get_density_map_gaussian
import json

dataset = 'B'
dataset_name = 'shanghaitech_part_{}'.format(dataset)
path = '../data-peng/original/shanghaitech/part_{}_final/test_data/images/'.format(dataset)
gt_path_csv = '../data-peng/original/shanghaitech/part_{}_final/test_data/ground_truth_csv/'.format(dataset)

if not os.path.exists(gt_path_csv):
    os.mkdir(gt_path_csv)

all_image_names = os.listdir(path)
all_image_names = [os.path.join(path,i) for i in all_image_names]
num_images = len(all_image_names)

with open("../json/new.json", "r") as f:
    test_data = json.load(f)

for i in range(1,num_images):
    if i%10 == 0:
        print('Processing %3d/%d files\n', i, num_images)

    input_img_name = all_image_names[i]
    print(input_img_name)
    im = cv2.imread(input_img_name)
    h, w, c = im.shape

    if c == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    annPoints = test_data[os.path.split(input_img_name)[-1]]
    annPoints = np.array(annPoints)
    h, w = im.shape
    im_density = get_density_map_gaussian(im,annPoints)
    np.savetxt(os.path.join(gt_path_csv, 'IMG_{}.csv'.format(((os.path.split(input_img_name)[-1]).split("_")[-1]).split(".")[0])), im_density, fmt='%10.5f', delimiter=",")
