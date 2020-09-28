import cv2
import numpy as np
import glob

# dict_images = dict()
# dict_images['Red'] = R
# dict_images['Green'] = G
# dict_images['Blue'] = B
# dict_images['Hue'] = H
# dict_images['Lightness'] = L
# dict_images['Saturation'] = S
# dict_images['Gray'] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
dict_images = dict()

keys = ['Red', 'Green', 'Blue', 'Hue', 'Lightness', 'Saturation', 'Gray']

thresholds = dict()
thresholds['Red'] = [150,200]
thresholds['Green'] = [0,255]
thresholds['Blue'] = [0,255]
thresholds['Hue'] = [0,255]
thresholds['Lightness'] = [0,255]
thresholds['Saturation']=[0, 255]
thresholds['Gray'] = [180, 255]

name_control_window = 'Control'

def onChange(x):
    pass

def trackbar():
    test_images = glob.glob('test_images/test*.jpg')[:3]
    
    for test_image in test_images:
        # img = cv2.cvtColor(cv2.imread(test_image),cv2.COLOR_BGR2RGB)
        img = cv2.imread(test_image)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        dict_image = dict()
        dict_image['Red'] = img[:,:,0]
        dict_image['Green'] = img[:,:,1]
        dict_image['Blue'] = img[:,:,2]
        dict_image['Hue'] = hls[:,:,0]
        dict_image['Lightness'] = hls[:,:,1]
        dict_image['Saturation'] = hls[:,:,2]        
        dict_image['Gray'] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_empty = np.zeros_like(dict_image['Gray'])
        img_merged = np.vstack((np.hstack((dict_image['Red'],dict_image['Green'],dict_image['Blue'])),
                                 np.hstack((dict_image['Hue'],dict_image['Lightness'],dict_image['Saturation'])),
                                 np.hstack((dict_image['Gray'], img_empty, img_empty))))
        img_to_show = cv2.resize(img_merged, dsize=(int(img_merged.shape[1]/3), int(img_merged.shape[0]/3)), interpolation=cv2.INTER_AREA)
        dict_images[test_image] = dict_image
        cv2.namedWindow(test_image)
        cv2.imshow(test_image, img_to_show)
        
    
    cv2.namedWindow(name_control_window)
    for key in keys:
        cv2.createTrackbar(key+'min', name_control_window, 0, 255, onChange)
        cv2.createTrackbar(key+'max', name_control_window, 0, 255, onChange)
        cv2.setTrackbarPos(key+'max', name_control_window, 255)
            
    while(True):
        if cv2.waitKey(1) & 0xFF == 27: # exit when 'Esc' Key is pressed
            break
        for key in keys:
            thresholds[key][0] = cv2.getTrackbarPos(key+'min', name_control_window)
            thresholds[key][1] = cv2.getTrackbarPos(key+'max', name_control_window)
        for test_image in test_images:
            dict_image = dict_images[test_image]
            dict_image_converted = dict()
            for key in keys:
                # img_converted = np.zeros_like(dict_image[key])
                # img_converted[(dict_image[key] > thresholds[key][0]) & (dict_image[key] <= thresholds[key][1])] = 1
                img_converted = dict_image[key].copy()
                img_converted = cv2.inRange(img_converted, thresholds[key][0], thresholds[key][1])
                ret, img_converted = cv2.threshold(img_converted, 0, 255,  cv2.THRESH_BINARY)
                dict_image_converted[key] = img_converted
            img_merged = np.vstack((np.hstack((dict_image_converted['Red'],dict_image_converted['Green'],dict_image_converted['Blue'])),
                                    np.hstack((dict_image_converted['Hue'],dict_image_converted['Lightness'],dict_image_converted['Saturation'])),
                                    np.hstack((dict_image_converted['Gray'], img_empty, img_empty))))
            img_to_show = cv2.resize(img_merged, dsize=(int(img_merged.shape[1]/3), int(img_merged.shape[0]/3)), interpolation=cv2.INTER_AREA)
            cv2.imshow(test_image, img_to_show)

        
    
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    trackbar()
    pass
