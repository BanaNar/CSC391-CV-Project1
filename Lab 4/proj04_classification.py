import numpy as np
from skimage.feature import hog
from sklearn.neighbors import NearestCentroid
import cv2
import os
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def readFiles(type):
    X = []
    if type == "palm":
        path = '/Users/eyangpc/PycharmProjects/StarterCode/palm/'

    elif type == "notpalm":
        path = '/Users/eyangpc/PycharmProjects/StarterCode/notpalm/'

    else:
        print('No such type!')

    for filename in os.listdir(path):
        X.append(cv2.imread(path + filename))

    return X


def hog_transform(mat, which):
    mat_hog = []
    if which == 'fd':
        for img in mat:
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, multichannel=True)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # img = img.resize((width, height), Image.ANTIALIAS)
            # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            # Xlbp.append(local_binary_pattern(img, n_points, radius, METHOD))
            mat_hog.append(fd)

    elif which == 'hog':
        for img in mat:
            fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, multichannel=True)
            mat_hog.append(hog_image)

    mat_hog = np.array(mat_hog)

    return mat_hog


# read images of palms and not palms
Xpalm = readFiles("palm")
Xnotpalm = readFiles("notpalm")

X = Xpalm + Xnotpalm
X = np.array(X)

y = []
for i in range(0, 800):
    y.append(1)

for i in range(0, 880):
    y.append(0)

# Xg = []
# for img in X:
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     Xg.append(img)
#

# FIXME
X_hog = hog_transform(X, 'fd')
# clf = SVC(kernel='rbf', gamma='scale')
# scores = cross_val_score(clf, X_hog, y, cv=5)
# print(scores)

# X_train_orig, X_test_orig, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=2019)

# #############################################################################

# X_train = hog_transform(X_train_orig, 'fd')
# X_test = hog_transform(X_test_orig, 'fd')

# ########## Support Vector Machine ###########################################

# print('--------- linear SVM ---------')
#
# clf = SVC(kernel='linear')
# clf = clf.fit(X_train, y_train)
# y_fit = clf.predict(X_test)
#
# print(classification_report(y_test, y_fit))
# print(confusion_matrix(y_test, y_fit))
#
print('--------- RBF SVM ---------')

# FIXME
clf = SVC(kernel='rbf', gamma='scale')
clf = clf.fit(X_hog, y)

# y_fit = clf.predict(X_test)
#
# print(classification_report(y_test, y_fit))
# print(confusion_matrix(y_test, y_fit))

# for i in range(0, len(y_test)):
#     if y_test[i] != y_fit[i]:
#         print(i, 'miss matched;',"expected:", y_fit[i])
#
#
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/18.jpg', X_test_orig[13])
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/32.jpg', X_test_orig[16])
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/59.jpg', X_test_orig[17])
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/61.jpg', X_test_orig[380])
# cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/mismatch/73.jpg', X_test_orig[382])


# #############################################################################

path = '/Users/eyangpc/PycharmProjects/StarterCode/fit/'
count = 1
plot_bk = []

for filename in os.listdir(path):
    img = cv2.imread(path + filename)
    print(path + filename)
    height, width = img.shape[:2]
    start_row, start_col = int(0), int(0)
    delta_height = int(100)
    delta_width = int(100)
    height_range = int(height/100)
    width_range = int(width/100)
    for i in range(0, height_range):
        for j in range(0, width_range):
            start_row, start_col = i * delta_height, j * delta_width
            end_row, end_col = (i + 1) * delta_height, (j + 1) * delta_width

            extract = img[start_row:end_row, start_col:end_col]
            # extract = np.array(extract)
            fd = hog(extract, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), multichannel=True)
            plot_bk.append(fd)


pred = clf.predict(plot_bk)
# count = 0
# mat = []
# for i in range(0, height_range):
#     row = []
#     for j in range(0, width_range):
#         row.append(pred[count])
#         count = count+1
#     mat.append(row)
#
# print(mat)

# #############################################################################

k = 0
for filename in os.listdir(path):
    img = cv2.imread(path + filename)
    print("Image drawing...")
    height, width = img.shape[:2]
    output = np.zeros(shape=(height, width, 3))
    start_row, start_col = int(0), int(0)
    delta_height = int(100)
    delta_width = int(100)
    height_range = int(height/100)
    width_range = int(width/100)
    for i in range(0, height_range):
        for j in range(0, width_range):
            start_row, start_col = i * delta_height, j * delta_width
            end_row, end_col = (i + 1) * delta_height, (j + 1) * delta_width

            extract = img[start_row:end_row, start_col:end_col]
            if pred[k] == 0:
                gray = cv2.cvtColor(extract, cv2.COLOR_RGB2GRAY)
                output[start_row:end_row, start_col:end_col] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            else:
                output[start_row:end_row, start_col:end_col] = extract

            k = k+1

cv2.imwrite('/Users/eyangpc/PycharmProjects/StarterCode/output.jpg', output)


