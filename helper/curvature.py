import numpy as np

ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

def get_middle(polynomes):
    avarage = polynomes[0] + polynomes[1] / 2.0
    return avarage

def compute_offset(img, polynomes):
    # eval for y is 720:
    middle = get_middle(polynomes)
    y = 720.0
    x = middle[0] * y ** 2 + middle[1] * y + middle[2]
    img_center = img.shape[1] / 2.0
    offset = img_center - x
    side = 'left'
    if offset < 0:
        side = 'right'
    offset_in_m = abs(offset)*xm_per_pix
    return offset_in_m, side

def compute_curvature(img, polynomes):
    # inspired from https://github.com/pkern90/CarND-advancedLaneLines :
    # basically take the polynome you have, and generate x,v values again
    # by evaluating the poynome at each pixel, then fit the polynome in metric space again
    middle = get_middle(polynomes)
    def eval(y):
        x = middle[0] * y ** 2 + middle[1] * y + middle[2]
        return x
    ys = np.linspace(0, img.shape[0]-1, img.shape[0])
    xs = map(eval, ys)
    y_eval = np.max(ys) * ym_per_pix
    fit_in_m = np.polyfit(ys * ym_per_pix, xs * xm_per_pix, 2)
    curverad = ((1 + (2 * fit_in_m[0] * y_eval + fit_in_m[1]) ** 2) ** 1.5) / np.absolute(2 * fit_in_m[0])
    return curverad