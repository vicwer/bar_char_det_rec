import os

file_path = open('../data/train_list/lbls_recg.txt', 'r')
path = open('../data/train_list/lbls.txt', 'w')
imgs = file_path.readlines()

for i in imgs:
    img = i.strip().split(' ')[0]
    path.write(img+'\n')
