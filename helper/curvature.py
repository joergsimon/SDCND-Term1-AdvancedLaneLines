import numpy as np
from collections import deque
from functools import reduce

ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

curvatures = deque([])
offsets = deque([])

def push_pop(queue, current):
    queue.append(current)
    if len(queue) > 10:
        queue.popleft()
    return queue

def get_middle(polynomes):
    avarage = polynomes[0] + polynomes[1] / 2.0
    return avarage

def compute_offset(img, polynomes):
    # eval for y is 720:
    middle = get_middle(polynomes)
    y = 720.0
    x = middle[0] * y ** 2 + middle[1] * y + middle[2]

    def eval_poly(p, y):
        x = p[0] * y ** 2 + p[1] * y + p[2]
        return x
    ys = np.array(np.linspace(0, img.shape[0]-1, img.shape[0]))
    left = np.array([eval_poly(polynomes[0], y) for y in ys])
    right = np.array([eval_poly(polynomes[1], y) for y in ys])

    # 10th value for x
    lane_middle = int((left[10] - right[10]) / 2.) + right[10]

    if (lane_middle - 640 > 0):
        leng = 3.66 / 2
        mag = ((lane_middle - 640) / 640. * leng)
        return mag, "Right"
    else:
        leng = 3.66 / 2.
        mag = ((lane_middle - 640) / 640. * leng) * -1
        return mag, "Left"
    # img_center = img.shape[1] / 2.0
    # offset = img_center - x
    # side = 'left'
    # if offset < 0:
    #     side = 'right'
    # offset_in_m = abs(offset)*xm_per_pix
    # push_pop(offsets, offset_in_m)
    # smoothed_offset_in_m = sum(offsets)/len(offsets)
    # return smoothed_offset_in_m, side

def compute_curvature(img, polynomes):
    # inspired from https://github.com/pkern90/CarND-advancedLaneLines :
    # basically take the polynome you have, and generate x,v values again
    # by evaluating the poynome at each pixel, then fit the polynome in metric space again
    middle = get_middle(polynomes)
    def eval_poly(y):
        x = middle[0] * y ** 2 + middle[1] * y + middle[2]
        return x
    ys = np.array(np.linspace(0, img.shape[0]-1, img.shape[0]))
    xs = np.array([eval_poly(y) for y in ys])
    y_eval = np.max(ys) * ym_per_pix
    fit_in_m = np.polyfit(ys * ym_per_pix, xs * xm_per_pix, 2)
    curverad = ((1 + (2 * fit_in_m[0] * y_eval + fit_in_m[1]) ** 2) ** 1.5) / np.absolute(2 * fit_in_m[0])

    push_pop(curvatures, curverad)
    smoothed_curverad = sum(curvatures) / len(curvatures)

    return curverad