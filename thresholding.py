import cv2
import numpy as np
import glob


img = cv2.cvtColor(cv2.imread(test_images[14]),cv2.COLOR_BGR2RGB)
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

dict_images = dict()
dict_images['Red'] = R
dict_images['Green'] = G
dict_images['Blue'] = B
dict_images['Hue'] = H
dict_images['Lightness'] = L
dict_images['Saturation'] = S
dict_images['Gray'] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

keys = ['Red', 'Green', 'Blue', 'Hue', 'Lightness', 'Saturation', 'Gray']

thresholds = dict()
thresholds['Red'] = (150,200)
thresholds['Green'] = (0,255)
thresholds['Blue'] = (0,255)
thresholds['Hue'] = (0,255)
thresholds['Lightness'] = (0,255)
thresholds['Saturation']=(0, 255)
thresholds['Gray'] = (180, 255)

def onChange(x):
    pass

def trackbar():
    test_images = glob.glob('test_images/samples_*.png')
    
    for test_image in test_images:
        cv2.imshow(test_image,img)        
        cv2.namedWindow(test_image)
        cv2.createTrackbar('B', test_image, 0, 255, onchange)
            
    while(True):
        b = cv2.getTrackbarPos('B', test_image)
        
            
