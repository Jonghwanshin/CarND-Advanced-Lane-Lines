import cv2
import numpy as np
import glob
from image_functions import *

thresholds_rgb = dict()
thresholds_rgb['Red'] = [0,255]
thresholds_rgb['Green'] = [0,255]
thresholds_rgb['Blue'] = [0, 255]
keys_rgb_filter = ['Red', 'Green', 'Blue']

thresholds_yellow = dict()
thresholds_yellow['Hue'] = [0,255]
thresholds_yellow['Saturation'] = [0,255]
thresholds_yellow['Value'] = [0, 255]
keys_yellow_filter = ['Hue', 'Saturation', 'Value']

thresholds_white = dict()
thresholds_white['Hue'] = [0,255]
thresholds_white['Lightness'] = [0,255]
thresholds_white['Saturation']= [0, 255]
keys_white_filter = ['Hue', 'Lightness', 'Saturation']

thresholds_sobel = dict()
thresholds_sobel['Sobel'] = [0,255]

name_control_window = 'Control'

def get_img_mask(img, dict_threshold):
    keys = dict_threshold.keys()
    min_threshold = np.array([dict_threshold[key][0] for key in keys])
    max_threshold = np.array([dict_threshold[key][1] for key in keys])
    # print(min_threshold, max_threshold)
    img_mask = cv2.inRange(img, min_threshold, max_threshold)
    return img_mask

def get_img_threshold(dict_threshold):
    keys = dict_threshold.keys()
    min_threshold = np.array([dict_threshold[key][0] for key in keys])
    max_threshold = np.array([dict_threshold[key][1] for key in keys])
    return [min_threshold, max_threshold]

def onChange(x):
    pass

def trackbar():
    test_image = 'test_images/test_sample.png'
    
    img_rgb = cv2.cvtColor(cv2.imread(test_image),cv2.COLOR_BGR2RGB)
    cv2.namedWindow(test_image)
    cv2.imshow(test_image, img_rgb)
        
    
    cv2.namedWindow(name_control_window)
    for key in keys_rgb_filter:
        cv2.createTrackbar(key+'min', name_control_window, 0, 255, onChange)
        cv2.createTrackbar(key+'max', name_control_window, 0, 255, onChange)
        cv2.setTrackbarPos(key+'max', name_control_window, 255)
    for key in keys_yellow_filter:
        cv2.createTrackbar(key+'Lmin', name_control_window, 0, 255, onChange)
        cv2.createTrackbar(key+'Lmax', name_control_window, 0, 255, onChange)
        cv2.setTrackbarPos(key+'Lmax', name_control_window, 0)
    for key in keys_white_filter:
        cv2.createTrackbar(key+'Wmin', name_control_window, 0, 255, onChange)
        cv2.createTrackbar(key+'Wmax', name_control_window, 0, 255, onChange)
        cv2.setTrackbarPos(key+'Wmax', name_control_window, 0)
    cv2.createTrackbar('sobel_min', name_control_window, 0, 255, onChange)
    cv2.createTrackbar('sobel_max', name_control_window, 0, 255, onChange)
            
    while(True):
        if cv2.waitKey(1) & 0xFF == 27: # exit when 'Esc' Key is pressed
            break
        for key in keys_rgb_filter:
            thresholds_rgb[key][0] = cv2.getTrackbarPos(key+'min', name_control_window)
            thresholds_rgb[key][1] = cv2.getTrackbarPos(key+'max', name_control_window)
        for key in keys_yellow_filter:
            thresholds_yellow[key][0] = cv2.getTrackbarPos(key+'Lmin', name_control_window)
            thresholds_yellow[key][1] = cv2.getTrackbarPos(key+'Lmax', name_control_window)
        for key in keys_white_filter:
            thresholds_white[key][0] = cv2.getTrackbarPos(key+'Wmin', name_control_window)
            thresholds_white[key][1] = cv2.getTrackbarPos(key+'Wmax', name_control_window)
        thresholds_sobel['Sobel'][0] = cv2.getTrackbarPos('sobel_min', name_control_window)
        thresholds_sobel['Sobel'][1] = cv2.getTrackbarPos('sobel_max', name_control_window)
        threshold_rgb = get_img_threshold(thresholds_rgb)
        threshold_yellow = get_img_threshold(thresholds_yellow)
        threshold_white = get_img_threshold(thresholds_white)
        threshold_sobel = get_img_threshold(thresholds_sobel)
        img_result = threshold_combined(img_rgb, 
                                        threshold_rgb,
                                        threshold_yellow, 
                                        threshold_white, 
                                        threshold_sobel)

        # ret, img_result = cv2.threshold(img_to_show, 1, 255, cv2.THRESH_BINARY)
        cv2.imshow(test_image, cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))

        
    
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    trackbar()
    pass
