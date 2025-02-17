## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

For details on the working of that project checkout the [report.md](./report.md)

---

The following files are important for the project:
* [report.md](./report.md) (report / project writeup)
* [report_images/](./report_images/) (folder with images and videos used for the report)
* [`examples/example.ipynb`](./examples/example.ipynb) (exploration of the algorithms based on single images)
* [`camera_cal/wide_dist_pickle.p`](./camera_cal/) (pickle file for configuration of undistortion. Computed by the notebook, used by `process.py`)
* [test_images/](./test_images/) (folder with images used by the notebook)
* [`process.py`](./process.py) (the script used to analyse / generate the video)
* [`helper/*.py`](./helper/) (python modules implementing the funcionality of the pipeline)
* [project_video.mp4](./project_video.mp4) (base video for later annotation)
* [result_video.mp4](./result_video.mp4) (final resulting video)

The video annotation is done with the script [`process.py`](./process.py). This needs a valid camera calibration pickle file, which can be generated by running the first cells of the notebook. [`process.py`](./process.py) has the values of the input video and the resulting video hardcoded. Also if you want to generate a video of an intermediate step exchange the function passed to `clip1.fl_image` by hand with one of the following:
* `process_image` -> generated the real video
* `process_image_mk_histogram` -> makes a video of the histogram used for search
* `process_image_mk_transf` -> makes a video of transformed binary images
* `process_image_mk_thresh` -> makes a video of the binary threshold images (looks nice ^_^)

In the report video on critical section is around 21 to 25s. To make a video of that section only uncomment the line:
`clip1 = VideoFileClip("project_video.mp4", audio=False).subclip(t_start=19, t_end=26)`
