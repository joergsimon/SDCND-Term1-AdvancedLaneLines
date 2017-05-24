from moviepy.editor import VideoFileClip
from helper.undistort import undistort, load_camera_calibration
from helper.threshold import threshold
from helper.transform import get_tranfrorm_matrix, transform
from helper.fit_polynomes import get_polynomes
from helper.curvature import compute_curvature
from helper.write_results import project_lanes, write_curvature

def process_image(image):
    global last_poly, camera_calibration
    undist = undistort(image, camera_calibration)
    thresh = threshold(undist)
    M, Minv = get_tranfrorm_matrix(thresh)  # currently this is hardcoded, but it might change
    transf = transform(thresh, M)
    polys, lb = get_polynomes(transf, last_poly)
    if not lb is None:
        last_poly = lb
    curv = compute_curvature(transf, polys)
    img_w_lanes = project_lanes(undist, polys, Minv)
    img_w_curv = write_curvature(img_w_lanes, curv)
    return img_w_curv

camera_calibration = load_camera_calibration()
last_poly = None
print("making a videofileclip object")
clip1 = VideoFileClip("/Users/joergsimon/Dropbox/uni/SDCND/SDCND-Term1-AdvancedLaneLines/project_video.mp4", audio=False)
print("--> start processing")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile("/Users/joergsimon/Dropbox/uni/SDCND/SDCND-Term1-AdvancedLaneLines/result_video.mp4", audio=False)
print("--> finished")