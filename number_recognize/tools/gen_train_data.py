# -*- coding: utf-8 -*-
"""
@author: kadua
"""

import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGE_DIR = '../coord_batch_05'
LABEL_DIR = '../data/train_list'
FONT_SIZE = 30
FIG_SIZE = 10
FIG_DPI = 60
MARGIN = 2
IMAGE_NUM = 40000

for d in [IMAGE_DIR]:
    if not os.path.isdir(d):
        os.makedirs(d)

def get_box(left, bottom, right, top):
    # from left bottom, clockwise
    return [
            #[left, bottom], 
            [left, top], 
            #[right, top], 
            [right, bottom]
    ]

def get_box_from_bbox(bbox, size):
    [[left, bottom], [right, top]] = bbox.get_points()
    return get_box(left, size - bottom, right, size - top)

def gen_labels(fname):
    txt = ''.join(random.sample('EHGFCDAB1234567890', np.random.randint(1, 5)))
    tname = fname[:-4] + txt + '.png'
    plt.figure(figsize=(FIG_SIZE,FIG_SIZE), dpi=FIG_DPI)
    txtobj = plt.text(0.03, 0.1, txt, size=FONT_SIZE)

    """
    plt.savefig(fname)
    [[left, top], [right, bottom]] = get_box_from_bbox(txtobj.get_window_extent(), FIG_SIZE*FIG_DPI)
    im = Image.open(fname).crop((left - MARGIN, top - MARGIN, right + MARGIN, bottom + MARGIN))
    im.save(fname)
    """

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = Image.frombytes('RGB', canvas.get_width_height(), 
                 canvas.tostring_rgb())
    [[left, top], [right, bottom]] = get_box_from_bbox(txtobj.get_window_extent(), FIG_SIZE*FIG_DPI)
    im = pil_image.crop((left - MARGIN, top - MARGIN, right + MARGIN, bottom + MARGIN))
    im.save(tname)
    plt.close('all')
    return tname

with open(os.path.join(LABEL_DIR, 'lbls_recg_05.txt'), 'w') as f:
    for i in tqdm(range(IMAGE_NUM)):
        fn = os.path.join(IMAGE_DIR, 'lbl05_' + str(i).zfill(7) + '_.png')
        tname = gen_labels(fn)
        f.write(tname+'\n')
