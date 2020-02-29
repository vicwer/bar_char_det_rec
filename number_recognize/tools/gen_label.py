import numpy as np
import os
import re

numbers = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, '$':18}

def one_hot_coding(classes_num, label):
    mask = np.zeros(classes_num, dtype='int32')
    mask[label] = 1
    return mask

def gen_label(filename, target_file):
    img_lists = open(filename, 'r')
    tf = open(target_file, 'w')
    img_names = img_lists.readlines()
    for i in img_names:
        name = re.findall(r'.*\_(.*)\.png', i.strip())[0]
        if len(name) < 4:
            name = name + int(4 - len(name)) * '$'
        labels = []
        for j in range(len(name)):
            numbers[name[j]]
            label = one_hot_coding(19, numbers[name[j]])
            labels.extend(label)
        labels = ' '.join([str(i) for i in labels])

        img_and_labels = i.strip() + ' ' + labels + '\n'
        tf.write(img_and_labels)

if __name__ == '__main__':
    filename = '../data/train_list/img_list.txt'
    target_file = '../data/train_list/train_list.txt'
    gen_label(filename, target_file)

