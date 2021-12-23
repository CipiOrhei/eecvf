import math
import os
# noinspection PyPackageRequirements
import cv2
import numpy as np


import config_main
from Utils.log_handler import log_setup_info_to_console, log_error_to_console, log_benchmark_info_to_console
from Benchmarking.Util.image_parsing import find_img_extension
from Benchmarking.Config.create_benchmark_job import set_gt_location, set_image_set, set_input_location, job_set


def spatial_frequency(img):
    """
    Spatial frequency (SF) is an image quality metric that measures the overall activity level in an image.
    SF= sqrt (RF^2 + CF^2), where RF is the row frequency and CF is the column frequency.
    https://ieeexplore.ieee.org/abstract/document/477498
    :param img: image to calculate SF on
    :return: SF value for image
    """
    M = img.shape[0]
    N = img.shape[1]
    img_shift_1_left = np.zeros(img.shape)
    img_shift_1_down = np.zeros(img.shape)

    img_shift_1_left[:, 1:] = img[:, :-1]
    img_shift_1_down[1:, :] = img[:-1, :]

    row_fq = math.sqrt((np.sum((img - img_shift_1_left) ** 2)) / (M * N))
    col_fq = math.sqrt((np.sum((img - img_shift_1_down) ** 2)) / (M * N))

    sp = math.sqrt(row_fq ** 2 + col_fq ** 2)

    return sp


# noinspection PyPep8Naming
def run_SF_benchmark(input_location: str, raw_image: str, jobs_set: list):
    """
    xxx
    :param input_location: location of algorithm images
    :param jobs_set: algo sets to evaluate
    :return: None
    """

    set_input_location(input_location)
    set_image_set(raw_image)
    job_set(jobs_set)

    run_CM_benchmark_SF()


def run_CM_benchmark_SF():
    """
    :return:
    """
    log_setup_info_to_console("BENCHMARKING CM SF")
    idx = 0

    for set in config_main.BENCHMARK_SETS:
        log_benchmark_info_to_console('Current set: {number}\{total} : {set}'.format(number=idx, total=len(config_main.BENCHMARK_SETS), set=set))
        idx += 1

        # try:
        if True:
            # Write results to disk
            results_path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, "SF")

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            csv = open(os.path.join(results_path, set + '.log'), "w+")
            csv.write('Per image (#, SF' + ':\n')
            # log_benchmark_info_to_console('Per image (#, RDE):\n')

            avg = 0
            count = 0

            for file in config_main.BENCHMARK_SAMPLE_NAMES:
                # find extension of images and gt_images
                if config_main.APPL_SAVE_JOB_NAME is True:
                    img_extension = find_img_extension(os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, set + '_' + file))
                else:
                    img_extension = find_img_extension(os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, file))

                if config_main.APPL_SAVE_JOB_NAME is True:
                    path_img_al = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, set + '_' + file + img_extension)
                else:
                    path_img_al = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set, file + img_extension)

                img_al = cv2.cvtColor(cv2.imread(path_img_al), cv2.COLOR_BGR2GRAY)

                try:
                    val = spatial_frequency(img_al)
                    avg += val
                    count += 1
                    csv.write('{:<10s} {:<10.6f}\n'.format(file, val))
                    # log_benchmark_info_to_console('{:<10s} {:<10.6f}\n'.format(file, val))
                except Exception as ex:
                    log_error_to_console("BENCHMARK CM SF: {file}".format(file=file), ex.__str__())

            log_benchmark_info_to_console('SF: {set:<10s} {cnt:<10.6f}\n'.format(set=set, cnt=avg / count))
            csv.write('SF: {set:<10s} {cnt:<10.6f}\n'.format(set=set, cnt=avg / count))

        # except Exception as ex:
        #     log_error_to_console('BENCHMARK CM RDEK' + int(k_value).__str__() + 'NOK', ex.__str__())


if __name__ == "__main__":
    pass
