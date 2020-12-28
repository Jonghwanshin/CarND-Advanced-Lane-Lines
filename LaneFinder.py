from image_functions import *
from LaneBoundary import LaneBoundary

NOT_FOUND = 0
FOUND = 1

# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 30/720 # meters per pixel in y dimension
XM_PER_PIX = 3.7/700 # meters per pixel in x dimension

class LaneFinder:
    """
        Finder for lane
    """
    def __init__(self, 
                 thres_err=0.01,
                 windows=9, 
                 margin_window=100, 
                 minpix=50, 
                 margin_search=100):
        self.__lane_left = []
        self.__lane_right = []
        self.__curvature = 0
        self.__vehicle_position = 0
        self.__status = NOT_FOUND
        self.__error_threshold = thres_err
        self.__windows = windows # nwindows: Choose the number of sliding windows
        self.__margin_window = margin_window # margin: Set the width of the windows +/- margin
        self.__minpix = minpix # minpix: Set minimum number of pixels found to recenter window
        self.__margin_search = margin_search # margin: Set the width of the windows +/- margin
        return
    
    def reset(self):
        self.__lane_left = []
        self.__lane_right = []
        self.__curvature = 0
        self.__vehicle_position = 0
        self.__status = NOT_FOUND
        return

    @property
    def lane_left(self):
        return self.__lane_left
    
    @property
    def lane_right(self):
        return self.__lane_right
    
    @property
    def curvature(self):
        return self.__curvature
    
    @property
    def vehicle_position(self):
        return self.__vehicle_position

    @property
    def error_threshold(self):
        return self.__error_threshold

    @error_threshold.setter
    def set_error_threshold(self, value):
        self.__error_threshold = value

    def find_lanes(self, img_input_warped):        
        """
            find lanes from an Bird-Eye View images
            img_input_warped: the image transformed to Bird-eye view
        """
        #img_binary_warped = threshold_combined(img_input_warped) #image thresholding

        if self.__status == NOT_FOUND: #the lane is not found on the previous image
            self.__status, left_lane_temp, right_lane_temp = self.fit_polynomial(img_input_warped)
        else:
            # search_around_poly if the lanes found on the previous frame
            self.__status, left_lane_temp, right_lane_temp = self.search_around_poly(img_input_warped)
        
        # if the lane is not found on the current frame predict from previous frame
        if (self.__status == NOT_FOUND):# or (error > self.error_threshold):
            left_lane_temp, right_lane_temp = self.predict_current_frame()
        
        self.__lane_left.append(left_lane_temp)
        self.__lane_right.append(right_lane_temp)

        self.__lane_left = self.__lane_left[-5:]
        self.__lane_right = self.__lane_right[-5:]

        # Calculate the radius of curvature in meters for both lane lines
        self.__curvature = self.measure_curvature_real()
        self.__vehicle_position = self.get_vehicle_pos(img_input_warped)
        return
    
    def visualize_lanes(self, img_input, 
                        each_lane=True,
                        center_lane=True,
                        sliding_window=True,
                        text=True):
        """
            Visualize lanes with images
        """
        warp_zero = np.zeros_like(img_input).astype(np.uint8)
        img_output = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, img_input.shape[0]-1, img_input.shape[0])

        left_fit = self.lane_left[-1].fit_coeffs
        right_fit = self.lane_right[-1].fit_coeffs
        
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        #print(pts_left.shape, pts_right.shape)
        pts = np.hstack((pts_left, pts_right))

        #print(pts)
        # Draw the lane onto the warped blank image
        try:
            cv2.fillPoly(img_output, np.int_([pts]), (0, 255, 0))
        except:
            print(pts.shape)
        
        # mark texts
        #print(self.__curvature, self.__vehicle_position)
        return img_output

    def find_lane_pixels(self, binary_warped):
        """
        find lane pixels with histogram approach
        """

        nwindows = self.__windows
        margin = self.__margin_window
        minpix = self.__minpix

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped, axis=0)
        
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

        return leftx, lefty, rightx, righty

    def fit_polynomial(self, binary_warped, nwindows=9, margin=100, minpix=50):
        # Find our lane pixels first
        status = FOUND
        leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)

        lane_left = LaneBoundary(leftx, lefty)
        lane_right = LaneBoundary(rightx, righty)
        
        #print(leftx[:5], lefty[:5], rightx[:5], righty[:5])
        for idx, lane in enumerate([lane_left, lane_right]):
            if not lane.fit_point():
                print('fitting for lane-{0} failed'.format(idx))
                print(lane.points_x.shape, lane.points_y.shape)
                status = NOT_FOUND

        return status, lane_left, lane_right
    
    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        status = FOUND

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_fit = self.__lane_left[-1].fit_coeffs
        right_fit = self.__lane_right[-1].fit_coeffs
        margin = self.__margin_search

        left_lane_inds = np.where(np.absolute(np.poly1d(left_fit)(nonzeroy)-nonzerox)< margin)[0]
        right_lane_inds = np.where(np.absolute(np.poly1d(right_fit)(nonzeroy)-nonzerox)< margin)[0]
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        lane_left = LaneBoundary(leftx, lefty)
        lane_right = LaneBoundary(rightx, righty)
        
        for idx, lane in enumerate([lane_left, lane_right]):
            if not lane.fit_point():
                print('fitting for lane-{0} failed'.format(idx))
                status = NOT_FOUND
        
        return status, lane_left, lane_right
    
    def get_vehicle_pos(self, img):
        """
            get vehicle position from the left and right curvature lane
        """
        # histogram between two image windows
        # TODO: change left_fit_cr, right_fit_cr
        left_fit_cr = self.lane_left[-1].fit_coeffs_meters
        right_fit_cr = self.lane_right[-1].fit_coeffs_meters
        y_bottom = img.shape[0]
        left_start_x = np.poly1d(left_fit_cr)(y_bottom)
        right_start_x = np.poly1d(right_fit_cr)(y_bottom)
        lane_center = (left_start_x + right_start_x) / 2
        img_center = img.shape[1] / 2
        vehicle_pos = (img_center - lane_center) * xm_per_pix
        return vehicle_pos

    def measure_curvature_real(self):
        """
            Calculates the curvature of polynomial functions in meters.
        """
        leftx = self.lane_left[-1].points_x 
        lefty = self.lane_left[-1].points_y
        rightx = self.lane_right[-1].points_x
        righty = self.lane_right[-1].points_y

        # Fit a second order polynomial to pixel positions in each lane line
        left_fit_cr = np.polyfit(lefty*YM_PER_PIX, leftx*XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(righty*YM_PER_PIX, rightx*XM_PER_PIX, 2)
        
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

    def predict_current_frame(self, mode='left'):
        """
            predict the current frame when not working
        """
        lane_left = self.__lane_left[-1]
        lane_right = self.__lane_right[-1]
        return lane_left, lane_right

