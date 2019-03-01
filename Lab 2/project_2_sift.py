import cv2
import numpy as np

img = cv2.imread('DSC_9259.JPG')
img_scale = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_scale = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

corners = cv2.cornerHarris(gray, 3, 3, 0.01)
kpsCorners = np.argwhere(corners > 0.01 * corners.max())
kpsCorners = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCorners]
kpsCorners, dscCorners = sift.compute(gray, kpsCorners)

cornersTwo = cv2.cornerHarris(gray_scale, 3, 3, 0.001)
kpsCornersTwo = np.argwhere(cornersTwo > 0.01 * cornersTwo.max())
kpsCornersTwo = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCornersTwo]
kpsCornersTwo, dscCornersTwo = sift.compute(gray_scale, kpsCornersTwo)

matchCorners = bfmatcher.match(dscCorners, dscCornersTwo)
matchCorners = sorted(matchCorners, key=lambda x:x.distance)
cornerMatch = cv2.drawMatches(img, kpsCorners, img_scale, kpsCornersTwo, matchCorners[:10], None, flags=2)

kp = sift.detect(gray, None)
kp, dsc = sift.compute(gray, kp)

kpTwo = sift.detect(gray_scale, None)
kpTwo, dscTwo = sift.compute(gray_scale, kpTwo)

matchSift = bfmatcher.match(dsc, dscTwo)
matchSift = sorted(matchSift, key=lambda x:x.distance)
siftMatch = cv2.drawMatches(img, kp, img_scale, kpTwo, matchSift[:10], None, flags=2)

cv2.imshow('Corner Match', cornerMatch)
cv2.imshow('Sift Match', siftMatch)

# End
key = cv2.waitKey(0);
cv2.destroyAllWindows();
