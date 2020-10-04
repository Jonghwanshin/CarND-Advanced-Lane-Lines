# **Finding Lane Lines on the Road** 

Project Goal
---

Implement an image pipeline that finds lane lines from road videos including:
* Takes road images from a video and return an annotated video stream
* Lane finding algorithm which:
  * finds the left and right lane lines and visualize the found lane
  * outputs visual display of the numerical estimation of lane curvature and vehicle position.


Reflection
---

### 1. Image Pipeline Description

The image pipeline is consists of the total of eight steps.

1) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images and apply a distortion correction to raw images.
2) Use color transforms, gradients, etc., to create a thresholded binary image.
3) Apply a perspective transform to rectify binary image ("birds-eye view").
4) Detect lane pixels and fit to find the lane boundary.
5) Determine the curvature of the lane and vehicle position with respect to center.
6) Warp the detected lane boundaries back onto the original image.
7) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**1) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images and apply a distortion correction to raw images**

First of all, it finds camera matrix from given chessboard images in `camera_cal` folder. The code for this step is in the first and second section in `01-distortion-correction.ipynb` file. Note that the size of the chessboard(`nx`,`ny`) must be considered before the computation. It find all chessboard corners with `cv2.findChessboardCorners()`. and finds calibration matrix and distortion coefficients with `cv2.calibrateCamera()` from found corners. Then, it applies undistortion to an image. It uses `cv2.undistort()` function for this step. I wraped this step with a function named `cal_undistort(img, mtx, dist)`, located at L26~28 of `image_functions.py`. An output example of this step is shown below.

![Undistorted Chessboard](./output_images/img_pipeline_undistorted_road.png)
*Fig 1. An example of undistorted road image*

**2) Use color transforms, gradients, etc., to create a thresholded binary image.**

After undistortion, it needs to seperate the lane from other objects in the image. The code for this step is in `02-color-gradient-thresholding.ipynb` and `thresholding.py`. First, I extracted test images from the input videos. Then, I aggregated the test images into a single image. Next, I created a simple OpenCV GUI app which can apply color and gradient thresholding into the test image to find out the best parameter to seperate lane lanes.

![Test image](./output_images/test_sample_combined.png)
*Fig 2. The thesholded test image with the best parameter I found*

After this step, I wraped the thresholding process with a function named `threshold_combined()` which located at L34-56 of `image_functions.py`. An example output image is shown below.



the input image to grayscale for further processing. It uses `cv2.cvtColor()` function for coverting images. Using grayscale image in computer vision has several advantages over using colored image including light intensity expression[^1]. In this project I used the grayscale image since I assumed that the brightness difference is the most crucial feature for distinguishing road surface and road lane. However, The image after this step is shown with somewhat greenish images in the above figure. This is because `matplotlib` uses color scheme for expressing a single-channel image.

**2) Gaussian Blur**

Second, it applies Gaussian Blur filter to reduce noises. It uses `cv2.GaussianBlur()` function for appling Gaussian Blur. The kernel size is set to 5 after few tries with the sample images. Note that the filter can cause the loss of important features from the image if the kernel size is too big. On the other hand, it doesn't filter out signal noises from the image if the kernel size is too small.

**3) Canny Edge Detection**

Third, it applies Canny Edge Detection to detecting any edges from image. It uses `cv2.CannyEdge()`  fot this processing step. I set the `low_threshold` to `50` and `high_threshold` to `150` after few tries with the sample images. The Canny Edge Detection algorithm uses a hyteresis threshold mechanism to find edge from a image. Therefore, an edge which the value is between the `low_threshold` and `high_threshold` is only detected if the edge connected to an edge with the value is higher than the `high_threshold`.

**4) Region Masking**

Forth, it applies region masking for seperating the lane and non-lane features. It retricts the region of interest in the shape of a trapezoid to consider the only the lane where the vehicle current at. It uses `cv2.fillPoly()` for generating image mask and `cv2.bitwise_and()` for region masking.

**5) Line Finding**

Then, it finds the left and the right road lane from the pre-processed image. It utilizes Hough Transform with `cv.HoughLinesP()` function to find straight lines. It returns a set of lines in the image. However, we need to combine those lines for each the left and the right road boundaries since visualizing all found lines may confuses the user. Therefore, I modified `draw_lines()` to combine the lines as shown below.

```python
def average_lane_line(lines, y_size):
    lines = np.array(lines)
    y_min = np.min(lines[:,3:])
    y_max = np.max(lines[:,3:])
    # get mean lines
    slope_mean = np.mean(lines[:,0])
    x1_mean = np.mean(lines[:,1])
    y1_mean = np.mean(lines[:,3])
    # get extreme points from line group
    y1 = int(y_min)
    x1 = int((y1 - y1_mean)/slope_mean + x1_mean)
    y2 = int(y_size) # the y value of the bottom of image
    x2 = int((y2 - y1_mean)/slope_mean + x1_mean)
    return (x1, y1), (x2, y2)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    left_line = []
    right_line = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            # slope get angle of cluster
            # cluster into 2 large segments
            slope = ((y2-y1)/(x2-x1))
            if abs(slope) > 0.3: # to get vertical lines
                if slope < 0:
                    left_line.append([slope,x1,x2,y1,y2])
                else:
                    right_line.append([slope,x1,x2,y1,y2]) 
    p1_left, p2_left = average_lane_line(left_line, img.shape[0])
    p1_right, p2_right = average_lane_line(right_line, img.shape[0])
    # add for left line
    cv2.line(img, p1_left, p2_left, color, thickness)
    # add for right line
    cv2.line(img, p1_right, p2_right, color, thickness)
```

The `draw_lines()` function divides the found lines into two groups, the left and right line. If the slope of the line is less than 0, it is classified as left, otherwise it is classified as right. Then, it represents each lane line groups by averaging slopes and intercepts of the lines and use the extreme points of the line groups. Finally, 
it overrays the found lane lines to the original image using `cv.addWeighted()` function.

**Additional Challenges**

The challenge video shows a driving scene on a curved road. Therefore, it needs to find curved lines for lane detection. I added modified region mask to remove ego vehicle hood, color picking with HSV image and RANSAC(Random Sample Consensus) algorithm for finding curved lines. 

Color Picking from HSV Image

This function converts a RGB Image to HSV image. Then, it picks the yellow and white area from the HSV image and apply yellow and white color mask to the original image. Then, it outputs a RGB image which has only white and yellow color.

```python
def color_pick(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    
    lower_yellow = np.array([10, 0, 0])
    upper_yellow = np.array([80, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)    
    return cv2.bitwise_and(image, image, mask = mask)
```

RANSAC polynomial fitting

This function inputs an edge image from canny edge detection and results the left and right lane lines from input image. I assumed that a curved line can be expressed with 2nd-order polynomial, which can also be expressed to a x^2 + bx + c =0. Therefore, I used RANSAC algorithm with 2nd-order to estimate the coefficients of the polynomial. I also added polynomial regression as safety measures for RANSAC algorithm. This process is applied to the left and right line edge to find both road boundaries.

```python
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def get_polylines(img, direction='left'):
    # get nonzero value from image
    np_nz = np.transpose(np.nonzero(img))
    
    xdata = np_nz[:,1]
    ydata = np_nz[:,0]
    
    # RANSAC estimator
    try:
        estimator = RANSACRegressor()
        model = make_pipeline(PolynomialFeatures(2), estimator)

        model.fit(xdata.reshape(-1,1), ydata)
        # Predict data of estimated models
        draw_x = np.arange(xdata.min(), xdata.max())
        draw_y = model.predict(draw_x.reshape(-1,1))
    except:
        f = np.poly1d(np.polyfit(xdata,ydata,2))
        draw_x = np.arange(xdata.min(), xdata.max())
        draw_y = f(draw_x)
    
    draw_points = np.asarray([draw_x, draw_y], dtype=np.int32).T
    return draw_points
```

Results

The algorithm can find solid curved line from the test video. However, it showed unstable behavior when it shows when line edge are weak, which are when line is in shadow and if the space between dashed line is too long.

### 2. Identify potential shortcomings with your current pipeline

The algorithm could not detect lane lines correctly on following road situations:
* Lane line splits and merge: entry ramps and exit ramps on highway
* Curve lane line: interchanges on highway, roundabouts
* Limited environmental conditions: heavy rain, low constrast

### 3. Suggest possible improvements to your pipeline

The limitations can be improved if:
* the algorithm can classify situations for line splits and merge with the angle between the found left and right lanes. or the number of existing lines at each line segments.
* the parameter for line finding algorithm(i.e. RANSAC) is properly tuned, and ensemble of two or more line finding algorithms.
* template matching could improve the performance of the line detection at limited environmental conditions. 

References
---
[^1]: "In image processing applications, why do we convert from RGB to Grayscale?", Quora, https://www.quora.com/In-image-processing-applications-why-do-we-convert-from-RGB-to-Grayscale.

[image1]: ./resources/CarND-P1-Fig1.png "Image Pipeline"