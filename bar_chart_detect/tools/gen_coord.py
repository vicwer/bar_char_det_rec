import numpy as np
import os
import re
import cv2

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [float(x), float(y), float(w), float(h)]

def load_file(file_path):
    '''
    load imgs_path, classes and labels
    '''
    imgs_path = []
    #classes = []
    #labels = []
    labels_and_classes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_path = '/diskdt/dataset/bar_chart_dataset/bar_chart_batch_1/' + line.strip().split(' ')[0]
            #cls = int(line.strip().split(' ')[1])
            cls_and_label = [float(i) for i in line.strip().split(' ')[1:]]
            if len(cls_and_label) > 30*5:
                continue
            cls_and_label = np.asarray(cls_and_label).reshape(-1,5)[:, [0,1,3,2,4]]
            cls_and_bb = []
            for i in range(cls_and_label.shape[0]):
                cls = [float(cls_and_label[i][0])]
                bb = convert((600,600), cls_and_label[i][1:])
                bb.extend(cls)
                cls_and_bb.extend(bb)

            if cls_and_label.shape[0] < 30:
                cls_and_bb = cls_and_bb + [0,0,0,0,0]*(30-int(cls_and_label.shape[0]))

            imgs_path.append(img_path)
            #classes.append(cls)
            #labels.append(label)
            #label.append(cls)
            labels_and_classes.append(cls_and_bb)
    return np.asarray(imgs_path), np.array(labels_and_classes)

file_path = '../data/train_list/train_list_tmp.txt'

imgs_path, labels_and_classes = load_file(file_path)

for i in range(imgs_path.shape[0]):
    img = cv2.imread(imgs_path[i])
    h, w = 600, 600
    coords = labels_and_classes[i].reshape(-1,5)

    for i in range(coords.shape[0]):
        coord = coords[i][:-1]
        centor_x = coord[0] * w
        centor_y = coord[1] * h
        box_w = coord[2] * w
        box_h = coord[3] * h
        x_left, x_right = int(centor_x - box_w / 2), int(centor_x + box_w / 2)
        y_top, y_bottom = int(centor_y - box_h / 2), int(centor_y + box_h / 2)

        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (0,0,0), 5)
    cv2.imshow('res', img)
    cv2.waitKey(0)
