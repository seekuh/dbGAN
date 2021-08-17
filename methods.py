import cv2
import numpy as np
import math
import os

############################################################
# methods for pertubing images AFTER generation 
############################################################

# vanilla pixelization with input imArray (RGB), block size b
# returns pixelated blur picture
def pixelization(imArray, b):
    h, w, channels = imArray.shape

    h1 = math.ceil(h/b)
    w1 = math.ceil(w/b)

    blur = cv2.resize(imArray, (w1,h1), interpolation = cv2.INTER_NEAREST)
    return blur

# vanilla gaussian blur with input imArray, kernel size k
    
def gaussian_blur(imArray,k):
    blur = cv2.GaussianBlur(imArray,(k,k), 0, borderType=cv2.BORDER_REPLICATE)
    return blur

# differentially private pixelization, additional params are: privacy
# budget epsilon, number of pixels to protect m. delta_p should be kept at 255.
# returns pixelated dp picture
def pixelization_dp(imArray, b, epsilon=0.5, m=16, delta_p = 255):  
    vals = imArray.shape

    # adapted to handle both grayscale and RGB
    if len(vals) ==2:
        h,w = vals
        channels = 1
    elif len(vals) ==3:
        h, w, channels = vals

    if m > h*w:  # if m is larger than image size
        m = h*w 
        
    h1 = math.ceil(h/b)
    w1 = math.ceil(w/b)

    # basic blur NP
    blur = cv2.resize(imArray, (w1,h1), interpolation = cv2.INTER_NEAREST)

    # DP blur
    loc, scale = 0, delta_p*m*channels/(b*b*epsilon)
    blur_dp = blur.copy()

    i=0
    j=0
    while i<h1:
        while j<w1:
            avg = blur_dp[i,j]
            s = np.random.laplace(loc, scale, channels)
            noisy_avg = avg + s
            for elem in np.nditer(noisy_avg, op_flags=['readwrite']):
            
                if elem > 255:
                    elem[...] = 255
                elif elem < 0:
                    elem[...] = 0
                else:
                    elem[...] = int(elem)
                    
            blur_dp[i,j] = noisy_avg 
            j=j+1
        i=i+1
        j=0

    return blur_dp

# differentially private gaussian blur, additional params are: privacy
# budget epsilon, number of pixels to protect m.

def gaussianblur_dp(imArray, k, b0, epsilon=0.5, m=16):
    vals = imArray.shape

    if len(vals) ==2:
        h,w = vals
        channels = 1
    elif len(vals) ==3:
        h, w, channels = vals


    # basic gaussian blur NP
    blur = cv2.GaussianBlur(imArray,(k,k), 0, borderType=cv2.BORDER_REPLICATE)

    
    # DP blur
    if m> h*w:
        m= h*w

    # pixelize, perturb, and then blur
    # b0 is intermidiate block size, should be a small value, eg, 2, 4, 6
    blur_dp = pixelization_dp(imArray, b0, epsilon, m)
     
    blur_dp = cv2.resize(blur_dp, (w,h), interpolation = cv2.INTER_NEAREST)

    gaussian_dp = cv2.GaussianBlur(blur_dp, (k,k), 0, borderType=cv2.BORDER_REPLICATE)

    return gaussian_dp

############################################################
# methods for pertubing images WHILE generation 
############################################################