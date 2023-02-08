import csv
from os import path

from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
import cv2
import numpy as np

basePath = "Dataset_2"


def rotate_image(image, max_angle=15):
    rotate_out = rotate(image, np.random.uniform(-max_angle, max_angle), mode='edge')
    return rotate_out


def translate_image(image, max_trans=5, height=32, width=32):
    translate_x = max_trans * np.random.uniform() - max_trans / 2
    translate_y = max_trans * np.random.uniform() - max_trans / 2
    translation_mat = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    trans = cv2.warpAffine(image, translation_mat, (height, width))
    return trans


def projection_transform(image, max_warp=0.8, height=32, width=32):
    # Warp Location
    d = height * 0.3 * np.random.uniform(0, max_warp)

    # Warp co-ordinates
    tl_top = np.random.uniform(-d, d)  # Top left corner, top margin
    tl_left = np.random.uniform(-d, d)  # Top left corner, left margin
    bl_bottom = np.random.uniform(-d, d)  # Bottom left corner, bottom margin
    bl_left = np.random.uniform(-d, d)  # Bottom left corner, left margin
    tr_top = np.random.uniform(-d, d)  # Top right corner, top margin
    tr_right = np.random.uniform(-d, d)  # Top right corner, right margin
    br_bottom = np.random.uniform(-d, d)  # Bottom right corner, bottom margin
    br_right = np.random.uniform(-d, d)  # Bottom right corner, right margin

    # Apply Projection
    transform = ProjectiveTransform()
    transform.estimate(np.array((
        (tl_left, tl_top),
        (bl_left, height - bl_bottom),
        (height - br_right, height - br_bottom),
        (height - tr_right, tr_top)
    )), np.array((
        (0, 0),
        (0, height),
        (height, height),
        (height, 0)
    )))
    output_image = warp(image, transform, output_shape=(height, width), order=1, mode='edge')
    return output_image


def transform_image(image, max_angle=15, max_trans=5, max_warp=0.8):
    height, width, channels = image.shape
    # Rotate Image
    rotated_image = rotate_image(image, max_angle)
    # Translate Image
    translated_image = translate_image(rotated_image, max_trans, height, width)
    # Project Image
    output_image = projection_transform(translated_image, max_warp, height, width)
    return (output_image * 255.0).astype(np.uint8)


def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with open(path.join(basePath, "Train.csv")) as f:
        lines = csv.reader(f)
        title = True
        for line in lines:
            if title:
                title = False
                continue
            # print(line)
            img = cv2.imread(path.join(basePath, line[7]))

            x, y = img.shape[0:2]
            show(cv2.resize(img, (200, 200)))
