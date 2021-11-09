import os
# noinspection PyPackageRequirements
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

import config_main
from Benchmarking.Util.image_parsing import find_img_extension
from Utils.log_handler import log_setup_info_to_console, log_benchmark_info_to_console, log_error_to_console


def mAP(img_gt, img):
    """
    Calculate the FOM for one image
    :param img_gt: ground truth image
    :param img: image to compare
    :return: FOM value
    """



def run_cbir_Top1(gt_location, show_per_file = False):
    """
    :return:
    """
    log_setup_info_to_console("CBIR ZuBuD Top1")
    idx = 1

    for set in config_main.BENCHMARK_SETS:
        log_benchmark_info_to_console('Current set: {}'.format(set))
        log_benchmark_info_to_console('Current set: {number}\{total}'.format(number=idx, total=len(config_main.BENCHMARK_SETS)))

        idx += 1

        try:
            # Write results to disk
            results_path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, 'ZuBuD Top1')

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            csv = open(os.path.join(results_path, set + '.log'), "w+")
            csv.write('Per image (#, ZuBuD Top1):\n')
            if show_per_file:
                log_benchmark_info_to_console('Per image (#, ZuBuD Top1):\n')

            Top1 = 0
            # count = 0
            list_missed = list()
            gt_dict = dict()

            for line in open(gt_location).readlines()[1:]:
                gt_dict['0' + line[:3]] = int(line[-4:-1])

            for file in config_main.BENCHMARK_SAMPLE_NAMES:
                # find extension of images and gt_images
                if config_main.APPL_SAVE_JOB_NAME is True:
                    path_img_al = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, set + '_' + file + '.txt')
                else:
                    path_img_al = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, file + '.txt')
                try:
                    list_inq_order = list()

                    for line in open(path_img_al).readlines():
                        value = line.split(' ')[-2]
                        value = int(value)
                        list_inq_order.append(value)

                    # val = gt_dict[file[-4:]] == list_inq_order[0] * 1
                    # Top1 += val
                    if gt_dict[file[-4:]] == list_inq_order[0]:
                        Top1 += 1

                        csv.write('{:<10s} {:<10.6f}\n'.format(file, 1))
                        if show_per_file:
                            log_benchmark_info_to_console('{:<10s} {:<10.6f}'.format(file, 1 * 100))
                    else:
                        list_missed.append(file[-4:])

                        csv.write('{:<10s} {:<10.6f}\n'.format(file, 0))
                        if show_per_file:
                            log_benchmark_info_to_console('{:<10s} {:<10.6f}'.format(file, 0 * 100))

                    # count += 1

                except Exception as ex:
                    log_error_to_console("BENCHMARK CBIR Top1: {file}".format(file=file), ex.__str__())

            log_benchmark_info_to_console('CBIR ZuBuD Top1: {:<10s} {:<10.6f}'.format(set, (Top1 / len(gt_dict)) * 100))
            log_benchmark_info_to_console('CBIR ZuBuD Top1 missed: {} '.format(list_missed.__str__()))
            csv.write('CBIR ZuBuD Top1: {:<10s} {:<10.6f}\n'.format(set, (Top1 / len(gt_dict)) * 100))

        except Exception as ex:
            log_error_to_console('BENCHMARK CBIR ZuBuD Top1 NOK', ex.__str__())



if __name__ == "__main__":
    pass
