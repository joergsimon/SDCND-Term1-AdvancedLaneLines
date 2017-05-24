import numpy as np

ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

def compute_curvature(img, polynomes):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fit = polynomes[0]
    right_fit = polynomes[1]
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    return (left_curverad, right_curverad)