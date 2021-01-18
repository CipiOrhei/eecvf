import os
import shutil
from scipy.io import loadmat, savemat
# noinspection PyPackageRequirements
import cv2
import numpy as np


def find_gt_from_files_copy(img_location, gt_folder, gt_new_folder):
    """
    Function finds the equivalent ground truth images from a file in the gt folder
    :param img_location: folder with input images
    :param gt_folder: gt images/mat folder
    :param gt_new_folder: folder where to copy
    :return: None
    """
    files = []
    gt_files = []

    for dirname, dirnames, filenames in os.walk(img_location):
        for filename in filenames:
            files.append(filename)
    print('files = ', files)
    print('len(files) = ', len(files))

    for dirname, dirnames, filenames in os.walk(gt_folder):
        for filename in filenames:
            gt_files.append(dirname + '/' + filename)
    print('gt_files = ', gt_files)
    print('len(gt_files) = ', len(gt_files))

    count = 0

    for file in files:
        for gt_file in gt_files:
            name = file.split('.')[0]
            gt_name = (gt_file.split('.'))[0].split('/')[-1]
            if name == gt_name:
                shutil.copy2(gt_file, gt_new_folder)
                print(gt_file)
                count += 1
    print('count = ', count)


def show_mat_picture(location, save_pict, show_image):
    images = loadmat(location, variable_names='groundTruth', appendmat=True).get('groundTruth')[0]

    h = len(images[0][0][0][0])
    w = len(images[0][0][0][0][0])
    full = np.zeros((h, w), np.float32)

    for i in range(len(images)):
        border_image_tmp = np.zeros((h, w), np.float32) + np.array(images[i][0][0][1])*255
        border_image_tmp = np.array(border_image_tmp, np.uint8)
        full += border_image_tmp

        if save_pict:
            image_name = str(location.split('\\')[-1]).split('.')[0] + '_' + str(i) + '.png'
            path = os.path.join(os.getcwd(), '../../', 'Logs', 'gt_images', image_name)
            cv2.imwrite(path, border_image_tmp)

        if show_image:
            cv2.imshow(str(i), border_image_tmp)

    full = cv2.normalize(full, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    if save_pict:
        image_name = str(location.split('\\')[-1]).split('.')[0] + '_' + 'all' + '.png'
        path = os.path.join(os.getcwd(), '../../', 'Logs', 'gt_images', image_name)
        cv2.imwrite(path, full)

    if show_image:
        cv2.imshow('all', full)

    if show_image:
        cv2.waitKey(500)


def save_gt_images_in_logs(folder):
    files = []

    for dirname, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            files.append(filename)

    print(files)

    for image in files:
        show_mat_picture(os.path.join(folder, image), True, False)


def create_mat_gt_bsds500_from_png(input_location, save_location):
    files = []

    for dirname, dirnames, filenames in os.walk(input_location):
        for filename in filenames:
            files.append(filename)

    for file in files:
        gt_image = cv2.cvtColor(cv2.imread(os.path.join(input_location, file)), cv2.COLOR_BGR2GRAY)//255
        gt = []
        gt.append({'Segmentation': np.zeros(gt_image.shape, dtype=np.dtype('H')), 'Boundaries': gt_image})

        savemat(os.path.join(save_location, file.split('.')[0] + '.mat'), mdict={'groundTruth': gt})


if __name__ == "__main__":
    create_mat_gt_bsds500_from_png(
        input_location='C:/repos/eecvf/TestData/CM_Dataset/gt_png',
        save_location='C:/repos/eecvf/TestData/CM_Dataset/gt_mat',
                                   )
