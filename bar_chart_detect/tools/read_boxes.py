import os
import re
import cv2
import numpy as np

source_path = '../data/train_list/train_list_tmp.txt'

with open(source_path, 'r') as f:
    while 1:
        line = f.readline()
        if not line:
            break
        img_path = re.split(' ', line)[0]
        coords = [int(float(i)) for i in re.split(' ', line.strip())[1:]]
        coords = np.asarray(coords).reshape(-1,5)
        print(coords.shape[0])
        img = cv2.imread('/diskdt/dataset/bar_chart_dataset/bar_chart_batch_1/'+img_path)
        h, w, _ = img.shape

        for i in range(coords.shape[0]):
            cv2.rectangle(img, (coords[i][1], coords[i][2]), (coords[i][3], coords[i][4]), (0,0,0), 4)
        cv2.imshow('img', img)
        cv2.waitKey(0)
