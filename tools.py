import numpy as np


def reduce_bounding_box(x, y, w, h, maximum_area):
    start_area = w*h
    if start_area <= maximum_area:
        return x, y, w, h
    shrink_proportion = np.sqrt((float(maximum_area)/float(start_area)))
    new_w = w * shrink_proportion
    new_h = h * shrink_proportion
    new_x = x + ((w - new_w) / 2.)
    new_y = y + ((h - new_h) / 2.)
    return int(np.round(new_x)), int(np.round(new_y)), int(np.round(new_w)), int(np.round(new_h))
