# -*- coding: utf-8 -*-
"""
@author: kadua
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGE_DIR = '../data'
LABEL_DIR = '../data'
FIG_SIZE = 10
FIG_DPI = 60
FONT_SIZE = 30

IMAGE_NUM = 10

colors = ['brown', 'red', 'green', 'blue', 'cyan', 'violet', 'orange', 'olive']

def plt_bar(fname, heights, barnames, title):
    assert len(barnames) <= 8 and len(barnames) >= 3
    np.random.shuffle(colors)
    # Choose the width of each bar and their positions
    #width = [0.1,0.2,3,1.5,0.3]
    #y_pos = [0,0.3,2,4.5,5.5]
    y_pos = np.arange(len(barnames))

    plt.figure(figsize=(FIG_SIZE,FIG_SIZE), dpi=FIG_DPI)
    rects = plt.bar(y_pos, heights, color=colors[:len(barnames)],  edgecolor='darkgray')#, width=width)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    # Add title and axis names
    #plt.title(title)
    #plt.xlabel('categories')
    #plt.ylabel('values')
     
    # Limits for the Y axis
    #plt.ylim(0,60)
    
    # Create names
    plt.xticks(y_pos, barnames)
    plt.savefig(fname)
    plt.close('all')
    return rects

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

def rect_to_pixel_pos(rect, size):
    x, y, w, h = rect.get_bbox().bounds
    x0, y0, x1, y1 = x, y, x + w, y + h
    x0, y0 = rect.figure.axes[0].transData.transform((x0, y0))
    x1, y1 = rect.figure.axes[0].transData.transform((x1, y1))
    return get_box(x0, size - y0, x1, size - y1)

def get_rect_tick_label_pos(rect, size):
    x_tick_label_pos = [get_box_from_bbox(x.get_window_extent(), size) for x in rect.axes.get_xmajorticklabels()]
    y_tick_label_pos = [get_box_from_bbox(y.get_window_extent(), size) for y in rect.axes.get_ymajorticklabels()[:-1]]
    return x_tick_label_pos, y_tick_label_pos

def flatten_list(ls):
    res = []
    if not isinstance(ls, list):
        res.append(ls)
        return res
    for x in ls:
        res += flatten_list(x)
    return res

def save_label(fname, bar_positions, xy_labels):
    with open(fname, 'w') as f:
        for img_index in range(len(bar_positions)):
            f.write('bar' + str(img_index) + '.png ')
            bars = bar_positions[img_index]
            bars_out = ' '.join('0 ' + ' '.join([str(int(x)) for x in flatten_list(bar)]) for bar in bars) + ' '
            f.write(bars_out)
            
            x_tick_label_pos, y_tick_label_pos = xy_labels[img_index]
            ticks = x_tick_label_pos + y_tick_label_pos
            tick_out = ' '.join('1 ' + ' '.join([str(int(x)) for x in flatten_list(t)]) for t in ticks)
            f.write(tick_out + '\n')

def plt_bars(n):
    bar_positions = []
    heights = []
    bar_names = []
    xy_labels = []
    for i in tqdm(range(n)):
        max_value = np.random.randint(20, 1000)
        bar_count = np.random.randint(3, 9)
        height = [np.random.randint(1, max_value) for i in range(bar_count)]
        bars = ('A', 'B', 'C', 'D', 'E', 'f', 'G', 'H')[:bar_count]
        fn = os.path.join(IMAGE_DIR, 'bar' + str(i) + '.png')
        rects = plt_bar(fn, height, bars, 'test bar ' + str(i))
        bar_position = [rect_to_pixel_pos(r, FIG_SIZE*FIG_DPI) for r in rects]

        bar_positions.append(bar_position)
        heights.append(height)
        bar_names.append(bars)
        xy_labels.append(get_rect_tick_label_pos(rects[0], FIG_SIZE*FIG_DPI))
    save_label(os.path.join(LABEL_DIR, 'detect.txt'), bar_positions, xy_labels)

plt_bars(IMAGE_NUM)

#rects = plt_bar('bar.png', [1,2,3], ['a', 'b', 'c'], 'test')

"""
names='groupA', 'groupB', 'groupC', 'groupD',
size=[12,11,3,30]

def plt_pie(fname, values, names, title):
    assert len(names) <= 8 and len(names) >= 3
    np.random.shuffle(colors)
    # create a figure and set different background
    plt.figure(figsize=(FIG_SIZE,FIG_SIZE), dpi=FIG_DPI)
    # Create a circle for the center of the plot
    my_circle=plt.Circle((0,0), .2, color='white')
    # Pieplot + circle on it
    plt.pie(values, labels=names, colors=colors[:len(names)])
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    # Label color
    plt.rcParams['text.color'] = 'black'
    plt.title(title)
    plt.savefig(fname)

plt_pie('pie.png', size, names, 'test pie')
"""

