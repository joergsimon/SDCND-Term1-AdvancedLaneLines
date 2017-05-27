# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

[//]: # (Image References)

[image1]: report_images/basic_thresh.png "Basic Binary Thresholds Grid"
[image2]: report_images/big_or_thresh.png "Combined Binary Thresholds Grid"
[image3]: report_images/masked_thresh.png "Masked Combined Binary Thresholds Grid"
[image4]: report_images/histogram.png "Histogram"
[image5]: report_images/chessboard.png "Chessboard"
[image6]: report_images/polynomes.png "fit polynomes"
[image7]: report_images/polynomes_search.png "polynomes used for search"
[image8]: report_images/transform_back.png "plynome transformed back"
[image9]: report_images/transformed.png "Transformed Thresholds"
[image10]: report_images/undistort_img.png "Undistored Image"
[video1]: ./project_video.mp4 "Video"

## Basic organization in the project

The project is organised in two parts: One [python notebook](examples/example.ipynb) was used for exploration and tuning the algorithm for a single image. Images in the report are generally taken from this notebook.

After experimenting with that a python script to analyse the video is created called `process.py`. The path to the video and output are hardcoded in the programm. To generate a video you have to change these values and run the script. It can also generate a video of the bin threshold, the warp or the histogram, if you change the function used in fl_image.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in ["./examples/example.ipynb"](examples/example.ipynb) 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Chessboard undistorted][image5]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Undistored car image][image10]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient and directional thresholds to generate a binary image. For the color threshold I transformed the image to the HLS color space and used the S channel additionally to the R channel in the RGB space and a greyscale image. I computed a magnitute threshold in the range of 30 to 100 based on the greyscale image. The greyscale image was also used for a directional threshold ranging from 0.7 to 1.3. A gradient in x direction was used for the red and grey image thresholded at 20 to 100. The S channel was directly thesholded on the pixel values between 170 to 255 to capture intense values. The result of this basic opeations can be seen in this picture:

![Basic binary threshold images][image1]

Based on these basic thresholds I explored combinations. In the end I used a big or over all these thresholds for a final threshold. Combinations can be seen in the next picture: 

![Combined thresholds with or][image2]

This image has a lot of noise in the background. However, this is not really a problem since in the perspective transform most of it goes out of the image range anyway. However, the we can still also apply the same masking we used in the project one, which I also did. The final image before transformation therefor looks this way:

![masked threshold image][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
