import math
import os
from math import log10
# noinspection PyPackageRequirements
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

import config_main
from Utils.log_handler import log_setup_info_to_console, log_error_to_console, log_benchmark_info_to_console
from Benchmarking.Util.image_parsing import find_img_extension
from Benchmarking.Config.create_benchmark_job import set_gt_location, set_image_set, set_input_location, job_set


def rde_calc(img, img_gt, k_value):
    """
    Dubuisson, M.P.; Jain, A.K. A modified Hausdorff distance for object matching. IEEE ICPR 1994, 1, 566-568
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.8155&rep=rep1&type=pdf
    :param img: edge map resulting of algorithm
    :param img_gt: ground truth image
    :return: psnr value for image
    """
    # calculate distances
    dist_gt = distance_transform_edt(np.invert(img_gt))
    dist_dc = distance_transform_edt(np.invert(img))

    # calculate sum(d^k(D))
    sum_dc = 0.0
    sum_gt = 0.0
    left = 0.0
    right = 0.0

    for i in range(0, img_gt.shape[0]):
        for j in range(0, img_gt.shape[1]):
            if img_gt[i, j]:
                sum_dc += dist_dc[i, j] ** k_value

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j]:
                sum_gt += dist_gt[i, j] ** k_value

    cn_cd = np.count_nonzero(img)
    cn_gt = np.count_nonzero(img_gt)

    if cn_cd != 0 :
        left = math.pow(sum_gt / cn_cd, 1.0/k_value)
    if cn_gt != 0:
        right = math.pow(sum_dc / cn_gt, 1.0/k_value)

    if cn_cd==0:
        rde = 1000
    else:
        rde = left + right

    return rde


# noinspection PyPep8Naming
def run_RDE_benchmark(input_location: str, gt_location: str,
                      raw_image: str, jobs_set: list,
                      k: int):
    """
    xxx
    :param input_location: location of algorithm images
    :param gt_location: location of gt images
    :param raw_image: location of raw images
    :param jobs_set: algo sets to evaluate
    :return: None
    """

    set_gt_location(gt_location)
    set_input_location(input_location)
    set_image_set(raw_image)
    job_set(jobs_set)

    run_CM_benchmark_RDE(k)


def run_CM_benchmark_RDE(k_value):
    """
    :return:
    """
    log_setup_info_to_console("BENCHMARKING CM RDEK" + int(k_value).__str__())
    idx = 0

    for set in config_main.BENCHMARK_SETS:
        log_benchmark_info_to_console('Current set: {number}\{total} : {set}'.format(number=idx, total=len(config_main.BENCHMARK_SETS), set=set))
        idx += 1

        # try:
        if True:
            # Write results to disk
            results_path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, "RDEK" + int(k_value).__str__())

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            csv = open(os.path.join(results_path, set + '.log'), "w+")
            csv.write('Per image (#, RDEK' + int(k_value).__str__() + ':\n')
            # log_benchmark_info_to_console('Per image (#, RDE):\n')

            avg = 0
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

                img_gt = cv2.cvtColor(cv2.imread(path_img_gt), cv2.COLOR_BGR2GRAY)
                img_al = cv2.cvtColor(cv2.imread(path_img_al), cv2.COLOR_BGR2GRAY)

                try:
                    val = rde_calc(img_al, img_gt, k_value)
                    avg += val
                    count += 1
                    csv.write('{:<10s} {:<10.6f}\n'.format(file, val))
                    # log_benchmark_info_to_console('{:<10s} {:<10.6f}\n'.format(file, val))
                except Exception as ex:
                    log_error_to_console("BENCHMARK CM RDEK{val}: {file}".format(val=int(k_value).__str__(), file=file), ex.__str__())

            log_benchmark_info_to_console('RDEK{val}: {set:<10s} {cnt:<10.6f}\n'.format(val=int(k_value).__str__(), set=set, cnt=avg / count))
            csv.write('RDEK{val}: {set:<10s} {cnt:<10.6f}\n'.format(val=int(k_value).__str__(), set=set, cnt=avg / count))

        # except Exception as ex:
        #     log_error_to_console('BENCHMARK CM RDEK' + int(k_value).__str__() + 'NOK', ex.__str__())


if __name__ == "__main__":
    pass
