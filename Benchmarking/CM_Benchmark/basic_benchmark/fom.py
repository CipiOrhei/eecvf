import os
# noinspection PyPackageRequirements
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

import config_main
from Benchmarking.Util.image_parsing import find_img_extension
from Utils.log_handler import log_setup_info_to_console, log_benchmark_info_to_console, log_error_to_console


def fom_calc(img_gt, img):
    """
    Calculate the FOM for one image
    :param img_gt: ground truth image
    :param img: image to compare
    :return: FOM value
    """
    # Compute the distance transform for the gt image .
    dist = distance_transform_edt(img_gt == 0)
    # constant
    alpha = 1.0 / 9.0
    fom = 0

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j]:
                fom += 1.0 / (1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(np.count_nonzero(img), np.count_nonzero(img_gt))

    return fom


def run_CM_benchmark_FOM():
    """
    The Pratt Figure of Merit (PFOM) is a method used to provide
    a quantitative comparison between edge detection algorithms
    in image processing
     W. K. Pratt, Digital Image Processing. New York: Wiley, 1977
    :return:
    """
    log_setup_info_to_console("BENCHMARKING CM FOM")
    idx = 0

    for set in config_main.BENCHMARK_SETS:
        log_benchmark_info_to_console('Current set: {}'.format(set))
        log_benchmark_info_to_console('Current set: {number}\{total}'.format(number=idx, total=len(config_main.BENCHMARK_SETS)))

        idx += 1

        try:
            # Write results to disk
            results_path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, 'FOM')

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            csv = open(os.path.join(results_path, set + '.log'), "w+")
            csv.write('Per image (#, FOM):\n')
            log_benchmark_info_to_console('Per image (#, FOM):\n')

            avg_fom = 0
            count = 0

            for file in config_main.BENCHMARK_SAMPLE_NAMES:
                # find extension of images and gt_images
                if config_main.APPL_SAVE_JOB_NAME is True:
                    img_extension = find_img_extension(os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, set + '_' + file))
                else:
                    img_extension = find_img_extension(os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, file))

                gt_extension = find_img_extension(os.path.join(config_main.BENCHMARK_GT_LOCATION, file))

                path_img_gt = os.path.join(config_main.BENCHMARK_GT_LOCATION, file + gt_extension)

                if config_main.APPL_SAVE_JOB_NAME is True:
                    path_img_al = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, set + '_' + file + img_extension)
                else:
                    path_img_al = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, file + img_extension)
                img_gt = cv2.imread(path_img_gt)
                img_al = cv2.imread(path_img_al)
                try:
                    val = fom_calc(cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_al, cv2.COLOR_BGR2GRAY))
                    avg_fom += val
                    count += 1
                    csv.write('{:<10s} {:<10.6f}\n'.format(file, val))
                    # log_benchmark_info_to_console('{:<10s} {:<10.6f}\n'.format(file, val * 100))
                except Exception as ex:
                    log_error_to_console("BENCHMARK CM FOM: {file}".format(file=file), ex.__str__())

            log_benchmark_info_to_console('FOM: {:<10s} {:<10.6f}\n'.format(set, (avg_fom / count) * 100))
            csv.write('FOM: {:<10s} {:<10.6f}\n'.format(set, (avg_fom / count) * 100))

        except Exception as ex:
            log_error_to_console('BENCHMARK CM FOM NOK', ex.__str__())


if __name__ == "__main__":
    pass
