import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Load and show my image
# Read image of puppy
img = cv2.imread('DSC_9259.JPG')
# # Resize by 1/2 if necessary
# small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
# Display resized image
# cv2.imshow("Resized", small)

# ========================== Spatial Filtering =============================================

# Define a square filter of any size k x k
# Create a 2-D box filter of size k x k and scale so that sum adds up to 1
size = 9
flt = np.random.randn(size, size)
flt = flt/flt.sum()

# Show the generated filter
plt.matshow(flt)
plt.show()

filterImg = np.zeros(img.shape, np.float64)  # array for filtered image
# Apply the filter to each channel
filterImg[:, :, :] = cv2.filter2D(img[:, :, :], -1, flt)
cv2.imshow('Filtered', filterImg.astype(np.uint8))
cv2.imwrite('img-flt.jpg',filterImg.astype(np.uint8))

# ========================== Smoothing, denoising and Edge Detection ========================

# Apply both Gaussian filter and Median filters to original image
# Change the (k, k) to adjust filter size
imgNoisy = cv2.imread('DSC_9259-0.40.JPG')
blur = cv2.GaussianBlur(imgNoisy,(3,3),0)
median = cv2.medianBlur(imgNoisy,3)

cv2.imshow('Original', imgNoisy)
cv2.imshow('Gaussian', blur)
cv2.imshow('Median', median)

# Apply canny to original puppy and noisy pussy
cannyImg = cv2.Canny(img, 50, 200)
cannyNoisy = cv2.Canny(imgNoisy, 200, 350) # If too much noise, increase value
cv2.imshow('Canny1', cannyImg)
cv2.imshow('Canny2', cannyNoisy)

# Apply canny to landscape image
land = cv2.imread('window-00-04.jpg')
cannyLand = cv2.Canny(land, 50, 240)
cv2.imshow('Canny Land', cannyLand)

# ========================== Frequency Analysis 4.1 ===========================================

# # Convert the image into grayscale image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grayImage.jpg", grayImg)
# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:grayImg.shape[0], 0:grayImg.shape[1]]

# Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients
F2_grayImg = np.fft.fft2(grayImg.astype(float))

Y = (np.linspace(-int(grayImg.shape[0]/2), int(grayImg.shape[0]/2)-1, grayImg.shape[0]))
X = (np.linspace(-int(grayImg.shape[1]/2), int(grayImg.shape[1]/2)-1, grayImg.shape[1]))
X, Y = np.meshgrid(X, Y)
# Plot the magnitude as image
plt.show()

# Plot the magnitude and the log(magnitude + 1) as images (view from the top)

# Standard plot: range of values makes small differences hard to see
magnitudeImage = np.fft.fftshift(np.abs(F2_grayImg))
magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
magnitudeImage = ski.img_as_ubyte(magnitudeImage)
cv2.imshow('Magnitude plot', magnitudeImage)

# Log(magnitude + 1) plot: shrinks the range so that small differences are visible
logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_grayImg)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
cv2.imshow('Log Magnitude plot', logMagnitudeImage)

# ========================== Frequency Analysis 4.2 ===========================================

# Calculate the index of the middle column in the image
col = int(grayImg.shape[1]/2)
# Obtain the image data for this column
#colData = smallNoisy[0:smallNoisy.shape[0], col, 0]
colData = img[0:grayImg.shape[0], col, 0]

# Plot the column data as a function
xvalues = np.linspace(0, len(colData) - 1, len(colData))
plt.plot(xvalues, colData, 'b')
#plt.savefig('/Users/paucavp1/Temp/function.png', bbox_inches='tight')
#plt.clf()
plt.show()

# Compute the 1-D Fourier transform of colData
F_colData = np.fft.fft(colData.astype(float))

# Plot the magnitude of the Fourier coefficients as a stem plot
# Notice the use off fftshift() to center the low frequency coefficients around 0
#xvalues = np.linspace(-int(len(colData)/2), int(len(colData)/2)-1, len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData)), 'g')
xvalues = np.linspace(0, len(colData), len(colData))
#markerline, stemlines, baseline = plt.stem(xvalues, (np.abs(F_colData)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.show()

# =========================== Frequency Filtering 5.1 =========================================

# Butterworth filters
# U and V are arrays that give all integer coordinates in the 2-D plane
#  [-m/2 , m/2] x [-n/2 , n/2].
# Use U and V to create 3-D functions over (U,V)
U = (np.linspace(-int(grayImg.shape[0]/2), int(grayImg.shape[0]/2)-1, grayImg.shape[0]))
V = (np.linspace(-int(grayImg.shape[1]/2), int(grayImg.shape[1]/2)-1, grayImg.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(X*X + Y*Y)
# create x-points for plotting
xval = np.linspace(-int(grayImg.shape[1]/2), int(grayImg.shape[1]/2)-1, grayImg.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.1 * D.max()

# The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# and all D(u,v) where D(u,v) > 0 equal to 0
idealLowPass = D <= D0

# Filter our small grayscale image with the ideal lowpass filter
# 1. DFT of image
print(grayImg.dtype)
FTgrayImg = np.fft.fft2(grayImg.astype(float))
# 2. Butterworth filter is already defined in Fourier space
# 3. Elementwise product in Fourier space (notice fftshift of the filter)
FTgrayImgFiltered = FTgrayImg * np.fft.fftshift(idealLowPass)
# 4. Inverse DFT to take filtered image back to the spatial domain
grayImgFiltered = np.abs(np.fft.ifft2(FTgrayImgFiltered))

# Save the filter and the filtered image (after scaling)
idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
grayImgFiltered = ski.img_as_ubyte(grayImgFiltered / grayImgFiltered.max())
cv2.imwrite("idealLowPass.jpg", idealLowPass)
cv2.imwrite("grayImageIdealLowpassFiltered.jpg", grayImgFiltered)

# Save gray ideal low pass image
grayImgLowPass = grayImgFiltered

# Plot the ideal filter and then create and plot Butterworth filters of order
# n = 1, 2, 3, 4
plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    # Create Butterworth filter of order n
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    # Apply the filter to the grayscaled image
    FTgrayImgFiltered = FTgrayImg * np.fft.fftshift(H)
    grayImgFiltered = np.abs(np.fft.ifft2(FTgrayImgFiltered))
    grayImgFiltered = ski.img_as_ubyte(grayImgFiltered / grayImgFiltered.max())
    cv2.imwrite("grayImageButterworth-n" + str(n) + ".jpg", grayImgFiltered)
    # cv2.imshow('H', H)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite("butter-n" + str(n) + ".jpg", H)
    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')
    if n == 1:
        grayImageButterLow1 = grayImgFiltered
    elif n == 2:
        grayImageButterLow2 = grayImgFiltered
    elif n == 3:
        grayImageButterLow3 = grayImgFiltered
    elif n == 4:
        grayImageButterLow4 = grayImgFiltered

# plt.show()
plt.savefig('butterworthFilters.jpg', bbox_inches='tight')

# =========================== Frequency Filtering 5.2 =========================================

# Ideal high pass filtered image
grayImgHighIdeal = grayImg - grayImgLowPass
cv2.imshow('grayImageHighPass', grayImgHighIdeal)
cv2.imwrite('grayImageHighPass.jpg', grayImgHighIdeal)

# Butterworth High Pass-n1
grayImgHighPass1 = grayImg - grayImageButterLow1
cv2.imshow('grayImageHighPass1', grayImgHighPass1)
cv2.imwrite('grayImageHighPass-n1.jpg', grayImgHighPass1)

# Butterworth High Pass-n2
grayImgHighPass2 = grayImg - grayImageButterLow2
cv2.imshow('grayImageHighPass2', grayImgHighPass2)
cv2.imwrite('grayImageHighPass-n2.jpg', grayImgHighPass2)

# Butterworth High Pass-n3
grayImgHighPass3 = grayImg - grayImageButterLow3
cv2.imshow('grayImageHighPass3', grayImgHighPass3)
cv2.imwrite('grayImageHighPass-n3.jpg', grayImgHighPass3)

# Butterworth High Pass-n4
grayImgHighPass4 = grayImg - grayImageButterLow4
cv2.imshow('grayImageHighPass4', grayImgHighPass4)
cv2.imwrite('grayImageHighPass-n4.jpg', grayImgHighPass4)

# End
key = cv2.waitKey(0);
cv2.destroyAllWindows();

