import cv2
import os
import numpy as np
import pickle
from skimage.transform import resize
from skimage.io import imread

model = pickle.load(open('./output_model/model.p', 'rb'))


def get_empty_or_not(spot_image):
    image = resize(spot_image, (15, 15, 3))
    image = np.array([image.flatten()])
    output = model.predict(image)

    return output == 0

def get_parking_spot_bounding_boxes(connected_components):
    spots = []
    coef = 1
    (total_labels, label_ids, values, centroid) = connected_components
    for i in range(1, total_labels):
        x = int(values[i, cv2.CC_STAT_LEFT]*coef)
        y = int(values[i, cv2.CC_STAT_TOP]*coef)
        w = int(values[i, cv2.CC_STAT_WIDTH]*coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT]*coef)

        spots.append([x, y, w, h])

    return spots

def get_data(data_dir):
    classes = os.listdir(data_dir)
    datas = []
    labels = []

    for class_name in classes:
        image_files = os.listdir(os.path.join(data_dir, class_name))
        for file in image_files:
            path = os.path.join(data_dir, class_name, file)
            image = imread(path)
            image = resize(image, (15,15))
            datas.append(image.flatten())
            labels.append(classes.index(class_name))
    datas = np.asarray(datas)
    labels = np.asarray(labels)
    return datas, labels, classes

def calc_diff(current, previous):
    return np.abs(np.mean(current) - np.mean(previous))

