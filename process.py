from moviepy.editor import VideoFileClip
from helper.undistort import undistort, load_camera_calibration
from helper.threshold import threshold
from helper.transform import get_tranfrorm_matrix, transform
from helper.fit_polynomes import get_polynomes
from helper.curvature import compute_curvature
from helper.write_results import project_lanes, write_curvature

import cv2
import numpy as np

def process_image_mk_thresh(image):
    global camera_calibration
    undist = undistort(image, camera_calibration)
    thresh = threshold(undist)
    thresh[thresh.nonzero()] = thresh[thresh.nonzero()]*255
    thresh = cv2.convertScaleAbs(thresh)
    img = np.zeros_like(image)
    img[:, :, 0] = thresh
    img[:, :, 1] = thresh
    img[:, :, 2] = thresh
    return img

def process_image_mk_transf(image):
    global camera_calibration
    undist = undistort(image, camera_calibration)
    thresh = threshold(undist)
    M, Minv = get_tranfrorm_matrix(thresh)  # currently this is hardcoded, but it might change
    transf = transform(thresh, M)
    img = np.zeros_like(image)
    transf[transf.nonzero()] = transf[transf.nonzero()]*255.0
    transf = cv2.convertScaleAbs(transf)
    img[:, :, 0] = transf
    img[:, :, 1] = transf
    img[:, :, 2] = transf
    return img

def process_image_mk_histogram(image):
    global camera_calibration
    undist = undistort(image, camera_calibration)
    thresh = threshold(undist)
    M, Minv = get_tranfrorm_matrix(thresh)  # currently this is hardcoded, but it might change
    transf = transform(thresh, M)
    histogram = np.sum(transf[transf.shape[0] // 2:, :], axis=0)
    scaled_hist = histogram * (720.0/max(histogram))
    basis = np.zeros_like(transf).astype(np.uint8)
    for i in range(len(scaled_hist)):
        basis[0:int(scaled_hist[i]), i] = 255
    img = np.zeros_like(image)
    img[:, :, 0] = basis
    img[:, :, 1] = basis
    img[:, :, 2] = basis
    return img

def process_image(image):
    global camera_calibration
    undist = undistort(image, camera_calibration)
    thresh = threshold(undist)
    M, Minv = get_tranfrorm_matrix(thresh)  # currently this is hardcoded, but it might change
    transf = transform(thresh, M)
    polys = get_polynomes(transf)
    curv = compute_curvature(transf, polys)
    img_w_lanes = project_lanes(undist, polys, Minv)
    img_w_curv = write_curvature(img_w_lanes, curv)
    return img_w_curv

camera_calibration = load_camera_calibration()
print("making a videofileclip object")
# for the final process the whole video
clip1 = VideoFileClip("project_video.mp4", audio=False)
# for debug only process the problematic area of 21...25s
# clip1 = VideoFileClip("project_video.mp4", audio=False).subclip(t_start=19, t_end=26)
print("--> start processing")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile("result_video-improved.mp4", audio=False)
print("--> finished")