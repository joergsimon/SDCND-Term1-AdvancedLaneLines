import numpy as np
import cv2

src = np.float32([
    (132, 703),
    (540, 466),
    (720, 466),
    (1147, 703)])

dst = np.float32([
    (200, 720),
    (200, 0),
    (1000, 0),
    (1000, 720)])


def get_tranfrorm_matrix(img):
    # for now, use the fixed values of src, dst. In future maybe load them dynamically
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def transform(img, M):
    warped = cv2.warpPerspective(img, M, img.shape[::-1], flags=cv2.INTER_LINEAR)
    return warped