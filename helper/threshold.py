from .threshold_utils import *

def threshold(img):
    l_channel, s_channel, gray, red = conversions(img)
    ksize = 3
    # basic thresholds:
    mag_binary = mag_thresh(gray, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
    red_gradx = abs_sobel_thresh(red, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grey_gradx = abs_sobel_thresh(red, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    s_binary = simple_threshold(s_channel, thresh=(170, 255))
    # combinations with direction thresh:
    dir_mag = np.zeros_like(dir_binary)
    dir_mag[(dir_binary == 1) & (mag_binary == 1)] = 1

    redx_mag = np.zeros_like(dir_binary)
    redx_mag[(dir_binary == 1) & (red_gradx == 1)] = 1

    greyx_mag = np.zeros_like(dir_binary)
    greyx_mag[(dir_binary == 1) & (grey_gradx == 1)] = 1

    sbin_mag = np.zeros_like(dir_binary)
    sbin_mag[(dir_binary == 1) & (s_binary == 1)] = 1
    # big or:
    big_or = np.zeros_like(dir_binary)
    big_or[(redx_mag == 1) | (greyx_mag == 1) | (dir_mag == 1) | (s_binary == 1)] = 1
    # mask:
    vertices = np.array(
        [[(0, big_or.shape[0]), (450, 325), (490, 325), (big_or.shape[1], big_or.shape[0])]],
        dtype=np.int32)
    img_masked = region_of_interest(big_or, vertices)
    return img_masked

def conversions(img):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    red = img[:, :, 0]
    return l_channel, s_channel, gray, red


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image