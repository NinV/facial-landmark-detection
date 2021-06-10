import json
import cv2
import os
import glob
import numpy as np 


f = open('data/cofw/annotations/cofw_test.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
meta_data = data['images']
annotations = data["annotations"]
image_paths = glob.glob('data/cofw/images/*')
for (i,j) in zip(meta_data,annotations):
    name_image = i['file_name']
    keypoints = j['keypoints']
    image = cv2.imread(f'data/cofw/images/{name_image}')
    points = np.array(keypoints).reshape(-1,3)
    print(len(points))
    for (x, y,_) in points:
        cv2.circle(image, (int(x + 0.5), int(y + 0.5)), radius=2, thickness=-1, color=[0, 255, 0])
        cv2.imshow('im',image)
        cv2.waitKey(0)
cv2.destroyAllWindows()

    
