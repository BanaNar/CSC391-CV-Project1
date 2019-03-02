import cv2
import numpy as np


def print_howto():
    print("""
        Control keys to change the video image:
            1. Canny Edge Detection - press 'c'
            2. Blurred Canny Edge Detection - press 'b'
            3. Harris Corner Detection - press 'h'
            4. SIFT - press 's'
            5. Quit - press 'q'
    """)


def edge_detect(img):

    # Adjust the high and low threshold
    edge = cv2.Canny(img, 100, 150) # FIXME Parameters
    # (100, 150)
    return edge


def corner_detect(img, gray_img):

    dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=5, k=0.06) # FIXME Parameters

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.02 * dst.max()] = [0, 0, 255]
    return img


def sift_image(img, gray_img):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6) # FIXME Parameters
    keypoints = sift.detect(gray_img, None)

    cv2.drawKeypoints(img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img


def blur(img):
    gaussian_blur = cv2.GaussianBlur(img, (9, 9), 0)  # FIXME Parameters
    return gaussian_blur


def rotate(img):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, int(0.5 * num_cols)],
                                     [0, 1, int(0.5 * num_rows)]])
    rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)
    img_translation = cv2.warpAffine(img, translation_matrix, (2 * num_cols,
                                                               2 * num_rows))
    img_rotation = cv2.warpAffine(img_translation, rotation_matrix,
                                  (num_cols * 2, num_rows * 2))
    return img_rotation


def scale(img):
    img_scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    return img_scaled


def translation(img):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols,
                                                               num_rows), cv2.INTER_LINEAR)
    return img_translation


# Corner Match
def corner_match(img1_harris, img2_harris):
    sift = cv2.xfeatures2d.SIFT_create()  # FIXME Parameters
    bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    img = img1_harris
    img_mod = img2_harris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_scale = cv2.cvtColor(img_mod, cv2.COLOR_BGR2GRAY)

    corners = cv2.cornerHarris(gray, 3, 3, 0.01)

    kpsCorners = np.argwhere(corners > 0.01 * corners.max())
    kpsCorners = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCorners]
    kpsCorners, dscCorners = sift.compute(gray, kpsCorners)

    cornersTwo = cv2.cornerHarris(gray_scale, 3, 3, 0.001)

    kpsCornersTwo = np.argwhere(cornersTwo > 0.01 * cornersTwo.max())
    kpsCornersTwo = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kpsCornersTwo]
    kpsCornersTwo, dscCornersTwo = sift.compute(gray_scale, kpsCornersTwo)

    matchCorners = bfmatcher.match(dscCorners, dscCornersTwo)
    matchCorners = sorted(matchCorners, key=lambda x: x.distance)
    cornerMatch = cv2.drawMatches(img, kpsCorners, img_mod, kpsCornersTwo, matchCorners[:10], None, flags=2)  # top 10 matched keypoints

    return cornerMatch


# SIFT Match
def sift_match(img1_sift, img2_sift):
    sift = cv2.xfeatures2d.SIFT_create()  # FIXME Parameters
    bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    img = img1_sift
    img_mod = img2_sift
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_scale = cv2.cvtColor(img_mod, cv2.COLOR_BGR2GRAY)

    kp = sift.detect(gray, None)
    kp, dsc = sift.compute(gray, kp)

    kpTwo = sift.detect(gray_scale, None)
    kpTwo, dscTwo = sift.compute(gray_scale, kpTwo)

    matchSift = bfmatcher.match(dsc, dscTwo)
    matchSift = sorted(matchSift, key=lambda x: x.distance)
    siftMatch = cv2.drawMatches(img, kp, img_mod, kpTwo, matchSift[:10], None, flags=2)

    return siftMatch


sift_img = cv2.imread('jpg.jpg')
sift_img_gray = cv2.cvtColor(sift_img, cv2.COLOR_BGR2GRAY)

# Part 2: SIFT Descriptors and Scaling
img0 = sift_image(sift_img, sift_img_gray)
img1 = sift_image(blur(sift_img), blur(sift_img_gray))
img2 = sift_image(rotate(sift_img), rotate(sift_img_gray))
img3 = sift_image(scale(sift_img), scale(sift_img_gray))
img4 = sift_image(translation(sift_img), translation(sift_img_gray))

cv2.imshow('origin_sift.jpg', img0)
cv2.imshow('blur_sift.jpg', img1)
cv2.imshow('rotate_sift.jpg', img2)
cv2.imshow('scale_sift.jpg', img3)
cv2.imshow('translation_sift.jpg', img4)

cv2.imwrite('origin_sift.jpg', img0)
cv2.imwrite('blur_sift.jpg', img1)
cv2.imwrite('rotate_sift.jpg', img2)
cv2.imwrite('scale_sift.jpg', img3)
cv2.imwrite('translation_sift.jpg', img4)

# Part 3: Keypoints and Matching
image1 = cv2.imread('cat.jpg') # Change input image
image2 = cv2.imread('cat.jpg') # Change input image
image3 = cv2.imread('l1.jpg') # Change input image
image4 = cv2.imread('l2.jpg') # Change input image

sift_match_img = sift_match(image1, image2)
corner_match_img = corner_match(image1, image2)
cv2.imshow('sift_match.jpg', sift_match_img)
cv2.imshow('corner_match.jpg', corner_match_img)
cv2.imwrite('sift_match.jpg', sift_match_img)
cv2.imwrite('corner_match.jpg', corner_match_img)

sift_rotate = sift_match(image1, rotate(image2))
corner_rotate = corner_match(image1, rotate(image2))
cv2.imshow('sift_rotate.jpg', sift_rotate)
cv2.imshow('corner_rotate.jpg', corner_rotate)
cv2.imwrite('sift_rotate.jpg', sift_rotate)
cv2.imwrite('corner_rotate.jpg', corner_rotate)

sift_scale = sift_match(image1, scale(image2))
corner_scale = corner_match(image1, scale(image2))
cv2.imshow('sift_scale.jpg', sift_scale)
cv2.imshow('corner_scale.jpg', corner_scale)
cv2.imwrite('sift_scale.jpg', sift_scale)
cv2.imwrite('corner_scale.jpg', corner_scale)

sift_blur = sift_match(image1, blur(image2))
corner_blur = corner_match(image1, blur(image2))
cv2.imshow('sift_blur.jpg', sift_blur)
cv2.imshow('corner_blur.jpg', corner_blur)
cv2.imwrite('sift_blur.jpg', sift_blur)
cv2.imwrite('corner_blur.jpg', corner_blur)

sift_transition = sift_match(image1, translation(image2))
corner_transition = corner_match(image1, translation(image2))
cv2.imshow('sift_transition.jpg', sift_transition)
cv2.imshow('corner_transition.jpg', corner_transition)
cv2.imwrite('sift_transition.jpg', sift_transition)
cv2.imwrite('corner_transition.jpg', corner_transition)

sift_angle = sift_match(image3, image4)
corner_angle = corner_match(image3, image4)
cv2.imwrite('sift_angle.jpg', sift_angle)
cv2.imwrite('corner_angle.jpg', corner_angle)

# # Get video image input
# print_howto()
# cap = cv2.VideoCapture(0)
#
# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")
#
# cur_mode = None
# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     c = cv2.waitKey(1)
#     if c & 0xFF == ord('q'):
#         break
#
#     if c != -1 and c != 255 and c != cur_mode:
#         cur_mode = c
#
#     # Set input keys
#     if cur_mode == ord('c'):
#         cv2.imshow('edge_detect', edge_detect(gray))
#
#     elif cur_mode == ord('b'):
#         cv2.imshow('blur_edge_detect', edge_detect(blur(frame)))
#
#     elif cur_mode == ord('h'):
#         cv2.imshow('corner_detect', corner_detect(frame, gray))
#
#     elif cur_mode == ord('s'):
#         cv2.imshow('sift_image', sift_image(frame, gray))
#
#     else:
#         cv2.imshow('video_img', frame)


# End
key = cv2.waitKey(0);
cv2.destroyAllWindows();