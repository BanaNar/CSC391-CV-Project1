import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D

# Load and show my image
# Read image of puppy and resize by 1/2
img = cv2.imread('DSC_9259.JPG')
small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
# Display resized image
cv2.imshow("Resized", small)

# ========================== Spatial Filtering =============================================

# # Define a square filter of any size k x k
# # Create a 1-D filter of length 3: [1/3, 1/3, 1/3] and apply to column data
# flt = np.zeros(small.shape, np.float64)
# size = 5
# flt = np.random.randn(size, size)
#
# # Create a 2-D box filter of default size 3 x 3 and scale so that sum adds up to 1
# w = np.ones((size, size), np.float64)
# w = np.random.randn(size, size)
# w = w / w.sum()
# # Display filter
# plt.matshow(w)
# plt.show()
# print(w)
#
# filterImg = np.zeros(flt.shape, np.float64)  # array for filtered image
# # Apply the filter to each channel
# filterImg[:, :, 0] = cv2.filter2D(small[:, :, 0], -1, w)
# filterImg[:, :, 1] = cv2.filter2D(small[:, :, 1], -1, w)
# filterImg[:, :, 2] = cv2.filter2D(small[:, :, 2], -1, w)
# cv2.imshow('Filtered', filterImg.astype(np.uint8))
# cv2.imwrite('img-flt.jpg',filterimg)

# ========================== Smoothing, denoising and Edge Detection ========================

# # Apply both Gaussian filter and Median filters to original image
# # Change the (k, k) to adjust filter size
# imgNoisy = cv2.imread('DSC_9259-0.40.JPG')
# blur = cv2.GaussianBlur(imgNoisy,(3,3),0)
# median = cv2.medianBlur(imgNoisy,3)
#
# cv2.imshow('Original', imgNoisy)
# cv2.imshow('Gaussian', blur)
# cv2.imshow('Median', median)
#
# # Apply canny to original puppy and noisy pussy
# cannyImg = cv2.Canny(img, 50, 200)
# cannyNoisy = cv2.Canny(imgNoisy, 200, 350) # If too much noise, can increase value
# cv2.imshow('Canny1', cannyImg)
# cv2.imshow('Canny2', cannyNoisy)
#
# # Apply canny to landscape image
# land = cv2.imread('window-00-04.jpg')
# cannyLand = cv2.Canny(land, 50, 240)
# cv2.imshow('Canny Land', cannyLand)

# ========================== Frequency Analysis ===============================================

# # Convert the image into grayscale image
# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cv2.imwrite("grayImage.jpg", grayImg)
# # create the x and y coordinate arrays (here we just use pixel indices)
# xx, yy = np.mgrid[0:grayImg.shape[0], 0:grayImg.shape[1]]
#
# # Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients
# F2_grayImg = np.fft.fft2(grayImg.astype(float))
# 
# Y = (np.linspace(-int(grayImg.shape[0]/2), int(grayImg.shape[0]/2)-1, grayImg.shape[0]))
# X = (np.linspace(-int(grayImg.shape[1]/2), int(grayImg.shape[1]/2)-1, grayImg.shape[1]))
# X, Y = np.meshgrid(X, Y)
#
# # Plot the magnitude as image
# plt.show()
#
# # Plot the magnitude and the log(magnitude + 1) as images (view from the top)
#
# # Standard plot: range of values makes small differences hard to see
# magnitudeImage = np.fft.fftshift(np.abs(F2_grayImg))
# magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
# magnitudeImage = ski.img_as_ubyte(magnitudeImage)
# cv2.imshow('Magnitude plot', magnitudeImage)
#
# # Log(magnitude + 1) plot: shrinks the range so that small differences are visible
# logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_grayImg)+1))
# logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
# logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
# cv2.imshow('Log Magnitude plot', logMagnitudeImage)










# End
key = cv2.waitKey(0);
cv2.destroyAllWindows();

