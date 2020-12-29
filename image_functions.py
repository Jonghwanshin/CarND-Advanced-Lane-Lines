import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

matFrontView = np.float32([[190, 720],
                      [577, 460],
                      [705, 460],
                      [1127, 720]])
matBirdEyeView = np.float32([[320, 720],
                      [320, 0],
                      [960, 0],
                      [960, 720]])

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


def show_images(img1, img2, title1, title2, filename='', save=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if save:
        output_filename_1 = 'output_images/{0}.png'.format(filename)
        plt.savefig(output_filename_1)


def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def threshold_combined(image, 
                       threshold_rgb=[(220, 0, 0),(255, 255, 0)],
                       threshold_yellow=[(17, 100, 100),(30, 255, 255)],
                       threshold_white=[(0, 0, 155),(255, 100, 255)], 
                       threshold_sobel_x=[30,255],
                       threshold_sobel_y=[30,255],
                       channels=3):
    """
    combined threshold to get binary image
    
    HYPERPARAMETERS
    * threshold rgb: the color filter to get images.
    * threhosld yellow: the color filter to get yellow lane lines.
    * threshold white: the color filter to get white lane lines.
    * threshold sobel: the sobel filter to get edges.
    """

    img_rgb = image.copy()
    img_hls = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2HLS)
    img_hsv = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2HSV)

    img_mask_overall = np.zeros_like(image)
    rgb_mask = cv2.inRange(img_rgb, threshold_rgb[0], threshold_rgb[1])
    yellow_mask = cv2.inRange(img_hsv, threshold_yellow[0], threshold_yellow[1])
    white_mask = cv2.inRange(img_hls, threshold_white[0], threshold_white[1])

    img_mask_overall = cv2.bitwise_or(rgb_mask, yellow_mask)
    img_mask_overall = cv2.bitwise_or(img_mask_overall, white_mask)
    img_yellow2 = cv2.bitwise_and(img_rgb, img_rgb, mask=rgb_mask)
    img_yellow = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask)
    img_white = cv2.bitwise_and(img_rgb, img_rgb, mask=white_mask)
    img_to_show = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_overall)
    
    img_yellow2 = cv2.cvtColor(img_yellow2, cv2.COLOR_RGB2GRAY)
    img_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_RGB2GRAY)
    img_white = cv2.cvtColor(img_white, cv2.COLOR_RGB2GRAY)
    img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_RGB2GRAY)

    color_binary = img_to_show
    
    #color_binary = im_bw

    # Sobel x
    sobelx = cv2.Sobel(color_binary, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))    
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= threshold_sobel_x[0]) & (scaled_sobel <= threshold_sobel_x[1])] = 255
    
    #img_hsv
    thres_adaptive = threshold_sobel_y[0]
    filter_size = threshold_sobel_y[1]
    if filter_size % 2 == 1 and filter_size > 1 and filter_size < 30:
        ##(thresh, im_bw) = cv2.threshold(img_hsv[:,:,1], th_min, th_max, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)
        im_bw = cv2.adaptiveThreshold(img_to_show, thres_adaptive, 
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, filter_size, 2)

        img_result = im_bw
    else:
        img_result = img_to_show

    del img_rgb
    if(channels == 1):
        return img_result
    else:
        return img_result #np.dstack((img_result, img_result, img_result))


def perspective_transform(img, src_mat, dst_mat):
    """
    perspective transform with src and dst matrix
    """
    M = cv2.getPerspectiveTransform(src_mat, dst_mat)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    find lane pixels with histogram approach
    
    HYPERPARAMETERS
    * nwindows: Choose the number of sliding windows
    * margin: Set the width of the windows +/- margin
    * minpix: Set minimum number of pixels found to recenter window
    """
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50, output=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50)

    # Fit a second order polynomial to each using `np.polyfit`
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        print('fitting left is failed {0} {0}'.format(lefty, leftx))
        left_fit = [0, 0, 0]
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        print('fitting right is failed {0} {0}'.format(righty, rightx))
        right_fit = [0, 0, 0]

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
#     # Colors in the left and right lane regions
    if output:
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

    return out_img, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Polynomial fit values from the previous frame
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = np.poly1d(left_fit)(ploty)
    right_fitx = np.poly1d(right_fit)(ploty)
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit


def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = np.where(np.absolute(np.poly1d(left_fit)(nonzeroy)-nonzerox)< margin)[0]
    right_lane_inds = np.where(np.absolute(np.poly1d(right_fit)(nonzeroy)-nonzerox)< margin)[0]
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window = np.array([np.transpose(np.vstack([left_fitx-50, ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx+50, ploty])))])
    line_pts = np.hstack((left_line_window, right_line_window))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
    result = window_img
    
    ## End visualization steps ##
    
    return result, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty

def translate_coordinates(ym_per_pix, xm_per_pix, leftx, rightx, lefty, righty):
    '''
    Generates fake data to use for calculating lane curvature.
    In your own project, you'll ignore this function and instead
    feed in the output of your lane detection algorithm to
    the lane curvature calculation.
    '''
    # Fit a second order polynomial to pixel positions in each lane line
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    return left_fit_cr, right_fit_cr

    
def measure_curvature_real(leftx, rightx, lefty, righty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''    
    left_fit_cr, right_fit_cr = translate_coordinates(ym_per_pix, xm_per_pix, leftx, rightx, lefty, righty)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(lefty)
    
    # the calculation of R_curve (radius of curvature)
    A_left = ym_per_pix * left_fit_cr[0]
    B_left = left_fit_cr[1] 
    left_curverad = np.power((1+(2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2), 3/2) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    curvature = (left_curverad + right_curverad)/2
    return curvature


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def put_info_to_img(img, curvature, vehicle_pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = [(100,100),(100,150),(100,200)]
    fontScale = 1.5
    fontColor = (255,255,255)
    lineType = 2
    
    text_radius = 'Radius of Curvature = {0:.1f}m'.format(curvature)
    text_vehicle_pos = 'Vehicle Pos = {0:.2f}m from the center'.format(vehicle_pos)
    cv2.putText(img, 
                text_radius, 
                bottomLeftCornerOfText[0], 
                font, fontScale, fontColor, lineType)
    cv2.putText(img, 
                text_vehicle_pos, 
                bottomLeftCornerOfText[1], 
                font, fontScale, fontColor, lineType)
    return img


def get_vehicle_pos(img, left_fit_cr, right_fit_cr):
    """
        get vehicle position from the left and right curvature lane
    """
    # histogram between two image windows
    y_bottom = img.shape[0]
    left_start_x = np.poly1d(left_fit_cr)(y_bottom)
    right_start_x = np.poly1d(right_fit_cr)(y_bottom)
    lane_center = (left_start_x + right_start_x) / 2
    img_center = img.shape[1] / 2
    vehicle_pos = (img_center - lane_center) * xm_per_pix
    return vehicle_pos

def visualize_lane(binary_warped, left_fitx, right_fitx, ploty):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    img_lane = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(img_lane, np.int_([pts]), (0,255, 0))
    return img_lane