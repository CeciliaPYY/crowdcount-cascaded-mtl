import json
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser(description="json transfer")
parser.add_argument("--json-path", dest="old_json", type=str)
parser.add_argument("--save-path", dest="new_json", type=str)
parser.add_argument("--image-path", dest="img_path", type=str)
parser.add_argument("--image-dest-path", dest="img_dest_path", type=str)
args = parser.parse_args()

with open(args.old_json, "r") as f1:
    data = json.load(f1)

for k, v in data.items():
    new_id = k.split(".")[0] + ".jpg"
    data[new_id] = data.pop(k)

new_data = {}

for k, v in data.items():
    data_num = len(v["regions"])
    if not os.path.isfile(args.img_path +"/"+ k):
        continue
    else:
        img = cv2.imread(args.img_path +"/"+ k)
        h, w = img.shape[:2]
        coor = np.zeros([data_num,2])
        for i in range(data_num):
            if h == 360*3 and w == 640*3:
                img = cv2.resize(img, [640,360])
                cv2.imwrite(args.img_path +"/"+ k, img)
                x = v["regions"][str(i)]["shape_attributes"]["cx"]/3
                y = v["regions"][str(i)]["shape_attributes"]["cy"]/3
                coor[i] = np.array([x, y])
            elif h == 360*6 and w == 640*6:
                img = cv2.resize(img, [640,360])
                cv2.imwrite(args.img_path +"/"+ k, img)
                x = v["regions"][str(i)]["shape_attributes"]["cx"]/6
                y = v["regions"][str(i)]["shape_attributes"]["cy"]/6
                coor[i] = np.array([x, y])

        new_data[k] = coor.tolist()

with open(args.new_json,"w") as f2:
    json.dump(new_data, f2)
