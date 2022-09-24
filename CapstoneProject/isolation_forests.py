import cv2
import numpy as np
import os
from imutils import paths
from sklearn.ensemble import IsolationForest


def color_distribution(img, bins = (4, 6, 3)):
    '''
    :param img: OpenCV-loaded image
    :param bins: 4 hue bins, 6 saturation bins, 3 value bins
    :return: histogram of the color distributions in the image
    '''

    histogram = cv2.calcHist([img], [0, 1, 2], None, bins, [1, 180, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()

    return histogram

def load_img_set(filepath, bins):
    '''
    :param filepath: filepath to the images
    :param bins: number of bins for the histogram
    :return: list containing histogram information for a file of images
    '''

    imagePaths = list(paths.list_images(filepath))
    data_list = []

    for imagePath in imagePaths:

        # load image and convert to HSV color space
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # quantify the image and update list
        image_features = color_distribution(image, bins)
        data_list.append(image_features)

    return np.array(data_list)


# load the image dataset with the given filepath to train the isolation forest
data = load_img_set('data/train/normal/', bins=(3, 3, 3))

# train model based on image dataset (98% normal images, 2% anomalous images)
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
model.fit(data)


def test_images(folder_path, folder_objects, type):
    '''
    :param folder_path: filepath location of the test images
    :param folder_objects: number of images in the folder
    :param type: image type (needed for file-naming conventions)
    :return:
    '''

    for i in range(1, folder_objects+1):

        filepath = folder_path + '\\' + type + '(' + str(i) + ').png'

        hsv = image = cv2.imread(filepath)
        cv2.imshow('image', image)
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = color_distribution(hsv, bins=(3, 3, 3))

        # use the anomaly detector model and extracted features to determine
        # if the example image is an anomaly or not
        preds = model.predict([features])[0]
        if preds == -1:
            print('anomaly')
        else:
            print('normal')


# test anomalalous images
folder_path_anomaly = 'data/test/anomaly/'
folder_objects_anomaly = len(os.listdir(folder_path_anomaly))
print('Testing Known Anomalous Images: ')
test_images(folder_path_anomaly, folder_objects_anomaly, 'anomaly')

# test normal images
folder_path_normal = 'data/test/normal/'
folder_objects_normal = len(os.listdir(folder_path_normal))
print('\nTesting Known Normal Images: ')
test_images(folder_path_normal, folder_objects_normal, 'normal')


# color distribution histogram for presentation
from matplotlib import pyplot as plt
img = cv2.imread('data/train/normal/normal (11).png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.ylim([0, 10000])
plt.show()
