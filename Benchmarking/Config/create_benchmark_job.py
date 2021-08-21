import glob
import os
import shutil

import config_main
from Benchmarking.CM_Benchmark.basic_benchmark.fom import run_CM_benchmark_FOM
from Benchmarking.CM_Benchmark.basic_benchmark.psnr import run_CM_benchmark_PSNR
from Benchmarking.CM_Benchmark.basic_benchmark.IoU import run_CM_benchmark_IoU
from Benchmarking.sb_benchmark.sb_IoU import run_SB_benchmark_IoU
from Benchmarking.bsds500.verify import run_verify_boundry
from Utils.log_handler import log_benchmark_info_to_console


def delete_folder_benchmark_out():
    """
    Delete the content of out folder where the saved images are.
    :return: None
    """
    path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, '*')
    files = glob.glob(path)
    for f in files:
        shutil.rmtree(f, ignore_errors=True)
    log_benchmark_info_to_console('DELETED CONTENT OF: {}'.format(config_main.BENCHMARK_RESULTS))


def set_gt_location(location: str):
    """
    Sets the location of ground truth of images
    :param location: location relative to repo
    :return: None
    """
    config_main.BENCHMARK_GT_LOCATION = location
    log_benchmark_info_to_console('GroundTruth location set: {}'.format(location))


def set_input_location(location: str):
    """
    Sets the location of input images
    :param location: location relative to repo
    :return: None
    """
    config_main.BENCHMARK_INPUT_LOCATION = location
    log_benchmark_info_to_console('Image location set: {}'.format(location))


def set_image_set(location: str):
    """
    Sets the location of raw images
    :param location: location relative to repo
    :return: None
    """
    image_set = []
    for dir_name, dir_names, file_names in os.walk(location):
        # array that stores all names
        for filename in file_names:
            image_set.append(filename.split('.')[0])

    config_main.BENCHMARK_SAMPLE_NAMES = image_set
    log_benchmark_info_to_console('Image set acquired: {}'.format(image_set))


def job_set(set_to_use: list):
    """
    Configure set of jobs
    :param set_to_use: list of jobs to add to job
    :return: None
    """
    config_main.BENCHMARK_SETS = set_to_use
    log_benchmark_info_to_console('Test sets: {}'.format(set_to_use))
    log_benchmark_info_to_console('Test sets number: {}'.format(len(set_to_use)))


def run_bsds500_boundary_benchmark(input_location: str, gt_location: str, raw_image: str,
                                   jobs_set: list, do_thinning: bool = True, px_offset=5):
    """
    Run BSDS boundary benchmark
    :param input_location: location of input of predicted edges
    :param gt_location: location of input of ground truth edges
    :param raw_image: location of input images
    :param do_thinning: if we want to do thinning of predicted edges
    :param px_offset: number of pixels offset between the ground truth and predicted edges to be considered as valid
    :param jobs_set: algo sets to evaluate
    :return: None
    """

    set_gt_location(location=gt_location)
    set_input_location(location=input_location)
    set_image_set(location=raw_image)
    job_set(set_to_use=jobs_set)

    if config_main.LINUX_OS:
        run_verify_boundry(thinning=do_thinning, max_distance_px=px_offset)
    else:
        log_benchmark_info_to_console("WE CAN'T RUN ON WINDOWS OS")


# noinspection PyPep8Naming
def run_PSNR_benchmark(input_location: str, gt_location: str,
                       raw_image: str, jobs_set: list):
    """
    T.B. Nguyen and D. Ziou, “Contextual and Non-Contextual Performance Evaluation of Edge Detectors,”
    Pattern Recognition Letters, vol. 21, no. 8, pp. 805-816, 2000.
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

    run_CM_benchmark_PSNR()


# noinspection PyPep8Naming
def run_FOM_benchmark(input_location: str, gt_location: str,
                      raw_image: str, jobs_set: list):
    """
    Run Figure of Merit benchmark
    W. K. Pratt, Digital Image Processing. New York: Wiley Interscience 1978
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

    run_CM_benchmark_FOM()


# noinspection PyPep8Naming
def run_IoU_benchmark(input_location: str, gt_location: str, class_list_name: list, raw_image: str, jobs_set: list, unknown_class: int,
                      is_rgb_gt=False, class_list_rgb_value=None, show_only_set_mean_value=False):
    """

    :param input_location: location of algorithm images
    :param gt_location: location of gt images
    :param class_list_name: list of name of classes
    :param unknown_class: which class is used for unknown predicted pixels
    :param is_rgb_gt: is the ground truth image is RGB colored
    :param class_list_rgb_value: list of greyscale values of classes
    :param raw_image: location of raw images
    :param jobs_set: algo sets to evaluate
    :param show_only_set_mean_value: show on console only the mean IoU for each set
    :return: None
    """

    set_gt_location(gt_location)
    set_input_location(input_location)
    set_image_set(raw_image)
    job_set(jobs_set)

    run_CM_benchmark_IoU(class_list_name, unknown_class, is_rgb_gt, class_list_rgb_value, show_only_set_mean_value)


# noinspection PyPep8Naming
def run_SB_IoU_benchmark(input_location: str, gt_location: str, jobs_set: list):
    """

    :param input_location: location of algorithm images
    :param gt_location: location of gt images
    :param class_list_name: list of name of classes
    :param unknown_class: which class is used for unknown predicted pixels
    :param is_rgb_gt: is the ground truth image is RGB colored
    :param class_list_rgb_value: list of greyscale values of classes
    :param raw_image: location of raw images
    :param jobs_set: algo sets to evaluate
    :param show_only_set_mean_value: show on console only the mean IoU for each set
    :return: None
    """

    set_gt_location(gt_location)
    set_input_location(input_location)
    job_set(jobs_set)

    run_SB_benchmark_IoU()


if __name__ == "__main__":
    pass
