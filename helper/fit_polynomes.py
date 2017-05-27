import numpy as np
from collections import deque
from functools import reduce

PI_thresh_approx = 3.141/15.0

last_poly = deque([])

def push_pop(last_poly, poly):
    last_poly.append(poly)
    if len(last_poly) > 10:
        last_poly.popleft()
    return last_poly

def get_polynomes(image):
    poly_is_ok = False
    lp = None
    if not len(last_poly) == 0:
        poly = find_poly(last_poly, image)
        poly_is_ok = is_ok(poly)
    if not poly_is_ok:
        print('find poly again')
        poly = find_poly_firstime(image, last_poly)
        poly_is_ok = is_ok(poly)
    if poly_is_ok:
        s1 = smoothe(last_poly[-1], poly) if len(last_poly) > 1 else poly
        push_pop(last_poly, s1)
        if len(last_poly) > 1:
            poly = reduce(smoothe, list(last_poly))
        return poly
    else: # poly is not ok even after searching again... return the last polynome for now...
        print('retain last')
        return last_poly[-1] if len(last_poly) > 0 else poly

def is_ok(poly):
    if poly is None:
        return False
    letf_poly = poly[0]
    right_poly = poly[0]
    vec1 = letf_poly/np.linalg.norm(letf_poly)
    vec2 = right_poly/np.linalg.norm(right_poly)
    angle = np.arccos(np.dot(vec1, vec2))
    return abs(angle) < PI_thresh_approx

def find_poly(last_poly, image):
    left_fit = last_poly[-1][0]
    right_fit = last_poly[-1][1]
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) == 0 or len(rightx)==0:
        return None
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # now we fit a polynome in the middle of the two to smoothe the curving

    #middle = (left_fit + right_fit)/2.0
    # so, now we only want to smooth the
    #left_fit[0] = (middle[0]+left_fit[0])/2.0
    #right_fit[0] = (middle[0]+right_fit[0])/2.0

    return (left_fit, right_fit)

def find_poly_firstime(image, last_poly):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(image.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        # if there was a good last polynome, bias the finding algorithm towards that polynome:
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

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return (left_fit, right_fit)

def smoothe(poly1, poly2):
    p1_left_fit  = poly1[0]
    p1_right_fit = poly1[1]
    p2_left_fit  = poly2[0]
    p2_right_fit = poly2[1]

    new_left  = (p1_left_fit + p2_left_fit) / 2.0
    new_right = (p1_right_fit + p2_right_fit) / 2.0
    return (new_left, new_right)