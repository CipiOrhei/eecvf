import os
from math import log10
# noinspection PyPackageRequirements
import cv2
import numpy as np

import config_main
from Utils.log_handler import log_setup_info_to_console, log_error_to_console, log_benchmark_info_to_console
from Benchmarking.Util.image_parsing import find_img_extension


def psnr_calc(img, img_gt):
    """
    calculate Peak Signal-to-Noise Ratio (PSNR)
    :param img: edge map resulting of algorithm
    :param img_gt: ground truth image
    :return: psnr value for image
    """
    psnr = 100
    mse = np.mean((img - img_gt) ** 2)

    if mse != 0:  # MSE is zero means no noise is present in the signal .
        max_pixel = np.max(img ** 2)
        psnr = 10 * log10(max_pixel / mse)
        # psnr = (max_pixel / mse)

    return psnr


def run_CM_benchmark_PSNR():
    """
    Run Peak Signal-to-Noise Ratio (PSNR) for a set of data
    :return:
    """
    # todo try using cv2.PSNR
    log_setup_info_to_console("BENCHMARKING CM PSNR")

    for set in config_main.BENCHMARK_SETS:
        log_benchmark_info_to_console('Current set: {}'.format(set))

        try:
            # Write results to disk
            results_path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, 'PSNR')

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            csv = open(os.path.join(results_path, set + '.log'), "w+")
            csv.write('Per image (#, PSNR):\n')
            log_benchmark_info_to_console('Per image (#, PSNR):\n')

            avg_psnr = 0
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
                    val = psnr_calc(img_al, img_gt)
                    avg_psnr += val
                    count += 1
                    csv.write('{:<10s} {:<10.6f}\n'.format(file, val))
                    log_benchmark_info_to_console('{:<10s} {:<10.6f}\n'.format(file, val))
                except Exception as ex:
                    log_error_to_console("BENCHMARK CM PSNR: {file}".format(file=file), ex.__str__())

            log_benchmark_info_to_console('PSNR: {:<10s} {:<10.6f}\n'.format(set, avg_psnr / count))
            csv.write('PSNR: {:<10s} {:<10.6f}\n'.format(set, avg_psnr / count))

        except Exception as ex:
            log_error_to_console('BENCHMARK CM PSNR NOK', ex.__str__())


if __name__ == "__main__":
    pass
