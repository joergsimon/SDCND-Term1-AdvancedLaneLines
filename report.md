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
[video1]: report_images/result_video-problem.mp4 "original algorithm, problematic section"
[video2]: report_images/result_video-problem-thresh.mp4 "original algorithm, problematic section, thresholds"
[video3]: report_images/result_video-problem-transformed.mp4 "original algorithm, problematic section, transformed"
[video4]: report_images/result_video-problem-histogram.mp4 "original algorithm, problematic section, histogram"
[video5]: report_images/result_video-improved.mp4 "improved algorithm, problematic section, histogram"
[video6]: ./project_video.mp4 "Video"

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

This calibration is actually calculated in the notebook and saved in a pickle file. The script `process.py` reads this pickle in `helper/undistort.py`

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

Beside the code in the notebook, the final functions used for the video can be found in `helper/threshold.py` and `helper/threshold_utils.py` respectively.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform was done using a fixed set of cooridnates more or less empirically choosen. It looks similar to the one provided in the example writeup, however, in the source image the rectangle stops a bit before the bottom to avoid having the car on the image, but still transforms that point to the bottom in the destination image. The final used coordinates can be found in `helper/transform.py`

For simplicity I assumed the video and images to have a fixed size, and just added the values. Of course, if the video format changes, this breaks everything.

```python
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
```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear relatively parallel in the warped image. In many images the car is a bit off center, so in a correct transform the lines do not have to be perfectly parallel, but at least almost.

![Transformed threshold image][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

All the code for this can be found in `helper/fit_polynomes.py`

The basic algorithm was a naive version pretty much like proposed in the lectures. The main logic is expressed in the code snipplet below:

```python
def get_polynomes(image):
    poly_is_ok = False
    if not len(last_polys) == 0:
        poly = find_poly(last_polys, image)
        poly_is_ok = is_ok(poly)
    if not poly_is_ok:
        print('find poly again')
        poly = find_poly_firstime(image, last_polys)
        poly_is_ok = is_ok(poly)
    if poly_is_ok:
        s1 = smoothe(last_poly[-1], poly) if len(last_polys) > 1 else poly
        push_pop(last_polys, s1)
        if len(last_polys) > 1:
            poly = reduce(smoothe, list(last_poly))
        return poly
    else: # poly is not ok even after searching again... return the last polynome for now...
        print('retain last')
        return last_poly[-1] if len(last_poly) > 0 else poly
```

So basically the last valid polynome is used for the search of the next one. If we do not have this last polynome, or the result of that search is not within a confidence (as determined by the angle of the polynomes, details later) we try to find a initial polynome using the histogram search outlined in the lectures. If after those steps a valid polynome can be found this polynome is avaraged with the last valid and pushed on the queue of valid polynomes. This queue always holds the last 10 valid polynomes. Then the Polynome is again avaraged over the last 10 valid polynomes. If no valid polynome could be found at all the last known valid polynome was used in the current frame. This occured very seldom.

If the polynomes of a frame are ok is determined by the angle between those two polynomes which indicates how paralell they are. The angle is computed by
```python
def is_ok(poly):
    if poly is None:
        return False
    letf_poly = poly[0]
    right_poly = poly[0]
    vec1 = letf_poly/np.linalg.norm(letf_poly)
    vec2 = right_poly/np.linalg.norm(right_poly)
    angle = np.arccos(np.dot(vec1, vec2))
    return abs(angle) < PI_thresh_approx
```
To indicate an ok result the similarity had to be below a threshold, choosen to be 1/15 of Pi by experimentation.

In the original algorithm the search for the polynomes used the histogram search and the polynome based search exactly like in the lecture.

This algorithm already performs quite ok for most of the track. However, at some areas of the track lightening condition and the type of concrete changes and introduce a lot of noise in this pipeline. An example of that is the time around 21s to 25s. The following video shows how the algorithm performed originally in this section:

[Problematic Section][video1]

To understand the problem I computed videos with the intermediate steps of the original algorithm on the section:

[Problematic Section, binary threshold video][video2]
[Problematic Section, perspective transformed video][video3]
[Problematic Section, histogram video][video4]

On the one side that showed me an programming error resulting in extreamly small values since I originally masked the binary image in a bit or with 255 values. On the other side especially the histogram video shows the problem the algorithm has to deal with. First there is a lot of noise. But second another area peaks as main peak in some frames, which is a problem when the search with the last polynome does not yield a valid result. To counter that beside the histogram search now also gets the last valid polinome as an input. When the histogram search is done for each window the x point of the lower y position of the window is computed and added as a bias to the mean. That means that a small peak can not move the line off center that fast. The imporant code section is the following:

```python
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        
        # .... ! SNIP ! that code is the same as in the original lectures
        
        # here is the interesting part:
        if len(good_left_inds) > minpix:
            correction = np.array([])
            if len(last_poly) > 0:
                left = last_poly[-1][0]
                center = left[0] * win_y_high ** 2 + left[1] * win_y_high + left[2]
                correction = np.array([center]*15)
            leftx_current = np.int(np.mean(np.concatenate((nonzerox[good_left_inds],correction))))
        if len(good_right_inds) > minpix:
            correction = np.array([])
            if len(last_poly) > 0:
                right = last_poly[-1][1]
                center = right[0] * win_y_high ** 2 + right[1] * win_y_high + right[2]
                correction = np.array([center]*15)
            rightx_current = np.int(np.mean(np.concatenate((nonzerox[good_right_inds],correction))))
```

This change, together with deciding on some parameters like the number of polynomes to keep and smooth and similar stuff improves the section quite dramatically (still with glitches) as seen in this video:

[Problematic Section, Improved algorithm][video5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This can be seen in the [notebook](examples/example.ipynb) at the bottom, or in the file `helper/curvature.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is a simple use of the inverse projection matrix. it can be seen at the bottom of the notebook or in the file `helper/transform.py` together with the main file where the inverse projection is done for the video.

![projected back][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

So the main issue was the occurance of short term peaks in the histogram when the search of the original polynome failed. I implemented a method who biases the polynome towards the last polynome also in the histogram search with windows. That could lead to problems f.e. if there is a sudden strong curve with a bad road condition. The bias lead that this curve might not be detected. Also I think crossing lines can be a problem.

What for shure can be done is instead of biasing running a peak detection algorithm over the histogram (something like trying to fit one, two or three gaussians or similar) and if more than one peak is detected take the most likely one based on the history. Also if no likely one exist track back the history even more. I guess that would perform better than my current algorithm. Additionally the speed of the car could be computed and with it how much of the last polynome must still be in the next frame and this information can be used for further search.
