import cv2

UD_CALIBRATION_FILE_DEFAULT = 'camera_cal/wide_dist_pickle.p'

def load_camera_calibration(filepath = UD_CALIBRATION_FILE_DEFAULT):
    import pickle
    dist_pickle = pickle.load(open(filepath, "rb"))
    return dist_pickle

def undistort(image, calibration):
    mtx = calibration["mtx"]
    dist = calibration["dist"]
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist