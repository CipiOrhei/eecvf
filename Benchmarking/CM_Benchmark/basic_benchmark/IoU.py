import os

# noinspection PyPackageRequirements
import cv2
import numpy as np

import config_main
from Utils.log_handler import log_setup_info_to_console, log_error_to_console, log_benchmark_info_to_console
from Benchmarking.Util.image_parsing import find_img_extension

"""
Intersection over union (IoU) calculation for evaluating an image segmentation model
https://en.wikipedia.org/wiki/Jaccard_index
"""


def mean_iou(labels, predictions, unknown_class, number_classes) -> list:
    """
    Calculate the IoU for one images. It returns a list of:
    %(unknown pixels/total pixels of predict) ... %(IoU for each class) ... % average IoU
    :param labels: ground truth image
    :param predictions: image to compare
    :param unknown_class: value of class that is used for unknown pixels
    :param number_classes: number of classes
    :return: IoU values
    """
    average_iou = 0.0
    seen_classes = 0
    list_iou_class = []

    for c in range(number_classes):
        if c != unknown_class:
            labels_c = (labels == c)
            predicted_c = (predictions == c)

            labels_c_sum = labels_c.sum()
            predicted_c_sum = predicted_c.sum()

            tmp = 1
            if (labels_c_sum > 0) or (predicted_c_sum > 0):
                seen_classes += 1

                intersect = np.logical_and(labels_c, predicted_c).sum()
                union = labels_c_sum + predicted_c_sum - intersect

                tmp = intersect / union
                average_iou += tmp
        else:
            predicted_c = (predictions == c)
            predicted_c_sum = predicted_c.sum()

            tmp = 0

            if predicted_c_sum > 0:
                tmp = predicted_c_sum / predictions.size

        list_iou_class.append(tmp)

    list_iou_class.append(average_iou / seen_classes if seen_classes else 0)

    return list_iou_class


# noinspection PyPep8Naming
def run_CM_benchmark_IoU(class_list_name: list, unknown_class, is_rgb_gt=False, class_list_rgb_value=None,
                         show_only_set_mean_value: bool = False) -> None:
    """
    Calculate the IoU for a set of images.
    :param class_list_name: list of names corresponding the classes
    :param unknown_class: the class value for unknown classification of pixels.
    :param is_rgb_gt: if ground truth images are in RGB format
    :param class_list_rgb_value: list of greyscale values of RGB colors used.
    :param show_only_set_mean_value: show on console only the mean IoU for each set
    :return: None
    """
    log_setup_info_to_console("BENCHMARKING CM IoU")

    for set_image in config_main.BENCHMARK_SETS:
        log_benchmark_info_to_console('Current set: {}'.format(set_image))

        try:
            # Write results to disk
            results_path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, 'IoU')

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            csv = open(os.path.join(results_path, set_image + '.log'), "w+")
            csv.write('Per image (#, IoU):\n')
            log_benchmark_info_to_console('Per image (#, IoU):\n')

            avg_fom = np.zeros((len(class_list_name) + 1), dtype=np.float)
            count = 0
            tmp = ''

            for el in range(len(class_list_name)):
                tmp += '{:<15s}\t'.format(class_list_name[el])

            csv.write('{:<15s} \t {:s} {:<20s}\n'.format('IoU: FILE', tmp, 'AVERAGE'))
            log_benchmark_info_to_console('IoU: {:<20s} \t {:s} {:<15s}\n'.format('FILE', tmp, 'AVERAGE'))

            for file in config_main.BENCHMARK_SAMPLE_NAMES:
                # find extension of images and gt_images
                if config_main.APPL_SAVE_JOB_NAME is True:
                    img_extension = find_img_extension(
                        os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set_image, set_image + '_' + file))
                else:
                    img_extension = find_img_extension(os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set_image, file))

                gt_extension = find_img_extension(os.path.join(config_main.BENCHMARK_GT_LOCATION, file))
                path_img_gt = os.path.join(config_main.BENCHMARK_GT_LOCATION, file + gt_extension)

                if config_main.APPL_SAVE_JOB_NAME is True:
                    path_img_al = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set_image, set_image + '_' + file + img_extension)
                else:
                    path_img_al = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set_image, file + img_extension)
                img_gt = cv2.imread(path_img_gt, cv2.IMREAD_GRAYSCALE)
                img_al = cv2.imread(path_img_al, cv2.IMREAD_GRAYSCALE)
                try:
                    if is_rgb_gt:
                        new_gt = np.zeros(img_gt.shape)

                        for el in range(len(class_list_rgb_value)):
                            new_gt += (img_gt == class_list_rgb_value[el]) * el

                        img_gt[:] = new_gt

                    class_list = mean_iou(img_gt, img_al, unknown_class, len(class_list_name))
                    count += 1
                    avg_fom += np.array(class_list)
                    tmp = ''

                    for el in range(len(class_list)):
                        tmp += '{:<15s}\t'.format('{:2.4f}'.format(class_list[el]))

                    csv.write('IoU: {:<20s} \t {:s}\n'.format(file, tmp))
                    if show_only_set_mean_value is False:
                        log_benchmark_info_to_console('IoU: {:<20s} \t {:s}\n'.format(file, tmp))
                except Exception as ex:
                    log_error_to_console("BENCHMARK CM IoU: {file}".format(file=file), ex.__str__())
            avg_fom = avg_fom / count

            tmp = ''
            for el in range(len(avg_fom)):
                tmp += '{:<15s}\t'.format('{:2.4f}'.format(avg_fom[el]))

            log_benchmark_info_to_console('IoU: {:<20s} \t {:s}\n'.format(set_image, tmp))
            csv.write('IoU: {:<20s} \t {:s}\n'.format(set_image, tmp))
        except Exception as ex:
            log_error_to_console('BENCHMARK CM IoU NOK', ex.__str__())
