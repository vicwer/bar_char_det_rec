import os
import re
import cv2

source_path = '../data/plate_list/gen_green_plate.txt'

with open(source_path, 'r') as f:
    while 1:
        line = f.readline()
        if not line:
            break
        img_path = re.split(' ', line)[0]
        coord = [float(i) for i in re.split(' ', line.strip())[2:]]
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        h, w = 600, 600

        centor_x = coord[0] * w
        centor_y = coord[1] * h
        box_w = coord[2] * w
        box_h = coord[3] * h
        x_left, x_right = int(centor_x - box_w / 2), int(centor_x + box_w / 2)
        y_top, y_bottom = int(centor_y - box_h / 2), int(centor_y + box_h / 2)
        #box_target.write(img_path + ' ' + str(x_left) + ' ' + str(y_top) + ' ' + str(x_right) + ' ' + str(y_bottom)+'\n')
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (0,255,255), 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
