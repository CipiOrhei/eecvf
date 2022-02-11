import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils

import cv2
import os
import sys

"""
TBD
"""

def prepare_dataset_labels(dataset_in, dataset_out, LabelMe_COLORS, LabelMe_BDT_CORRELATION, BDT_COLORS, BDT_CLASSES):
    """
    Please download https://github.com/cvjena/labelmefacade for this experiment
    This function will correlate the images from LabelMe Facade to TMBuD dataset
    :param dataset_in: folder of dataset to process
    :param dataset_out: folder where to save the processed dataset
    :param LabelMe_COLORS: original labels in RGB
    :param LabelMe_BDT_CORRELATION: correlated labels to change in binary mode
    :param BDT_COLORS: correlated labels to change to RGB format
    :param BDT_CLASSES: correlated class labels to change of RGB format
    :return: None
    """
    Application.set_input_image_folder(dataset_in)
    Application.set_output_image_folder(dataset_out)
    Application.do_get_image_job(port_output_name='RAW')

    Application.do_class_correlation(port_input_name='RAW', port_output_name='BDT_LABELS', class_list_in=LabelMe_COLORS, class_list_out=LabelMe_BDT_CORRELATION)
    Application.do_class_correlation(port_input_name='BDT_LABELS', port_output_name='BDT_LABELS_PNG', class_list_in=BDT_CLASSES, class_list_out=BDT_COLORS)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()


def prepare_dataset_img(dataset_out, dataset_img_input):
    """
    Copy dataset to certain location
    :param dataset_out: input location
    :param dataset_img_input: output location
    :return:
    """
    Application.set_input_image_folder(dataset_img_input)
    Application.set_output_image_folder(dataset_out)
    Application.do_get_image_job(port_output_name='RAW')
    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()


def prepare_dataset_TMBuD(TMBuD_CORRELATION):
    """
    Please download https://github.com/CipiOrhei/TMBuD for this experiment
    This function will correlate the images from TMBuD Facade to TMBuD dataset
    :param TMBuD_CORRELATION: class labels to correlate to
    :return: None
    """
    # change this for you use case
    dataset_input_labels_tmbud = r'c:\repos\eecvf_git\TestData\TMBuD\parsed_dataset\SEMSEG_EVAL_FULL\label_full\TEST\classes'
    dataset_processed_tmbud = 'Logs/TMBuD/labels'

    Application.set_input_image_folder(dataset_input_labels_tmbud)
    Application.set_output_image_folder(dataset_processed_tmbud)
    Application.do_get_image_job(port_output_name='RAW', direct_grey=False)

    Application.do_class_correlation(port_input_name='RAW', port_output_name='BDT_LABELS', class_list_in=[0, 1, 2, 3, 4, 5, 6, 7], class_list_out=TMBuD_CORRELATION)
    Application.do_class_correlation(port_input_name='BDT_LABELS', port_output_name='BDT_LABELS_PNG', class_list_in=BDT_CLASSES, class_list_out=BDT_COLORS)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()


def main_training_data(height, width, data_input_img):
    """
    Preparing the data for training the semantic segmentation
    :param height: height of output image
    :param width: width of output image
    :param data_input_img: location of input images
    :return: None
    """
    Application.set_output_image_folder('Logs/application_results_ml_raw')
    Application.set_input_image_folder(data_input_img)
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW')
    list_of_ports_to_move = list()
    list_of_ports_to_move.append(Application.do_resize_image_job(port_input_name='RAW', new_height=height, new_width=width, is_rgb=True, interpolation=cv2.INTER_CUBIC, port_output_name='RAW_RESIZE'))
    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    for el in range(len(list_of_ports_to_move.copy())):
        list_of_ports_to_move[el] += '_LC0'
    Application.create_folders_from_list_ports(folder_names=['Logs/ml_exchange/TRAIN_INPUT', 'Logs/ml_results/VAL_INPUT'], list_port=list_of_ports_to_move, folder_ratios=[0.5, 0.5])

    Application.set_input_image_folder('Logs/ml_exchange/TRAIN_INPUT')
    Application.set_output_image_folder('Logs/application_results_ml_raw')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')

    list_of_ports_to_move = list()
    list_of_ports_to_move.append(Application.do_resize_image_job(port_input_name='RAW', new_height=height, new_width=width, is_rgb=True, interpolation=cv2.INTER_CUBIC, port_output_name='RAW_RESIZE'))
    list_of_ports_to_move.append(Application.do_flip_image_job(port_input_name='RAW_RESIZE', is_rgb=True, flip_horizontal=True, flip_vertical=False, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_zoom_image_job(port_input_name='RAW_RESIZE', is_rgb=True, zoom_factor=1.05, do_interpolation=True,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_zoom_image_job(port_input_name='RAW_RESIZE', is_rgb=True, zoom_factor=1.1, do_interpolation=True,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_rotate_image_job(port_input_name='RAW_RESIZE', is_rgb=True, angle=10, reshape=False, extend_border=True,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_rotate_image_job(port_input_name='RAW_RESIZE', is_rgb=True, angle=-10, reshape=False, extend_border=True,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_gaussian_blur_noise_job(port_input_name='RAW_RESIZE', is_rgb=True, mean_value=0.01, variance=0.01,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_gaussian_blur_noise_job(port_input_name='RAW_RESIZE', is_rgb=True, mean_value=0.05, variance=0.05,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_salt_pepper_noise(port_input_name='RAW_RESIZE', is_rgb=True, density=0.1, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_salt_pepper_noise(port_input_name='RAW_RESIZE', is_rgb=True, density=0.4, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_gaussian_blur_image_job(port_input_name='RAW_RESIZE', is_rgb=True, sigma=1.4, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_gaussian_blur_image_job(port_input_name='RAW_RESIZE', is_rgb=True, sigma=3, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_mean_blur_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel_size=9, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_mean_blur_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel_size=7, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_motion_blur_filter_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel_size=7, angle=90,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_motion_blur_filter_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel_size=7, angle=-90,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=1.5,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, beta=10,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=1.5, beta=10,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=2.5, beta=10,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=0.5, beta=0, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=0.1, beta=50, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_pixelate_image_job(port_input_name='RAW_RESIZE', is_rgb=True, nr_pixels_to_group=3,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_pixelate_image_job(port_input_name='RAW_RESIZE', is_rgb=True, nr_pixels_to_group=2,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_sharpen_filter_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_unsharp_filter_expanded_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=0.5, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_unsharp_filter_expanded_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=0.8, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    for el in range(len(list_of_ports_to_move.copy())):
        print(list_of_ports_to_move[el])
        list_of_ports_to_move[el] += '_LC0'

    Application.create_folder_from_list_ports(folder_name='Logs/ml_results/TRAIN_INPUT', list_port=list_of_ports_to_move)

    Utils.close_files()


def main_training_label(height, width, dataset_input):
    """
    Preparing the data for training the semantic segmentation
    :param height: height of output image
    :param width: width of output image
    :param dataset_input: location of input images
    :return: None
    """
    Application.set_output_image_folder('Logs/application_results_ml_labels')
    Application.set_input_image_folder(dataset_input)
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW_LABEL')
    Application.do_grayscale_transform_job(port_input_name='RAW_LABEL', port_output_name='GREY_LABEL')
    list_of_ports_to_move = list()

    list_of_ports_to_move.append( Application.do_resize_image_job(port_input_name='GREY_LABEL', new_height=height, new_width=width, is_rgb=False, interpolation=cv2.INTER_NEAREST, port_output_name='RAW_RESIZE'))

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    for el in range(len(list_of_ports_to_move.copy())):
        list_of_ports_to_move[el] += '_LC0'

    Application.create_folders_from_list_ports(folder_names=['Logs/ml_exchange/TRAIN_LABELS', 'Logs/ml_results/VAL_LABEL'],
                                               list_port=list_of_ports_to_move, folder_ratios=[0.5, 0.5])

    Application.set_input_image_folder('Logs/ml_exchange/TRAIN_LABELS')
    Application.set_output_image_folder('Logs/application_results_ml_labels')
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW_LABEL')
    Application.do_grayscale_transform_job(port_input_name='RAW_LABEL', port_output_name='GREY_LABEL')

    list_of_ports_to_move = list()
    list_of_ports_to_move.append(Application.do_resize_image_job(port_input_name='GREY_LABEL', new_height=height, new_width=width, is_rgb=False, interpolation=cv2.INTER_NEAREST, port_output_name='RAW_RESIZE'))
    list_of_ports_to_move.append(Application.do_flip_image_job(port_input_name='RAW_RESIZE', is_rgb=False, flip_horizontal=True, flip_vertical=False, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_zoom_image_job(port_input_name='RAW_RESIZE', is_rgb=False, zoom_factor=1.05, do_interpolation=False, port_output_name='ZOOM_1.05_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_zoom_image_job(port_input_name='RAW_RESIZE', is_rgb=False, zoom_factor=1.1, do_interpolation=False, port_output_name='ZOOM_1.1_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_rotate_image_job(port_input_name='RAW_RESIZE', is_rgb=False, angle=10, reshape=False, extend_border=True, do_interpolation=False, port_output_name='ROTATE_ANGLE_10_BORDER_EXT_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_rotate_image_job(port_input_name='RAW_RESIZE', is_rgb=False, angle=-10, reshape=False, extend_border=True, do_interpolation=False, port_output_name='ROTATE_ANGLE_-10_BORDER_EXT_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_gaussian_blur_noise_job(port_input_name='RAW_RESIZE', is_rgb=False, mean_value=0, variance=0, port_output_name='GAUSS_NOISE_MEAN_VAL_0_01_VAR_0_01_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_gaussian_blur_noise_job(port_input_name='RAW_RESIZE', is_rgb=False, mean_value=0, variance=0, port_output_name='GAUSS_NOISE_MEAN_VAL_0_05_VAR_0_05_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_salt_pepper_noise(port_input_name='RAW_RESIZE', is_rgb=False, density=0, port_output_name='S&P_NOISE_DENS_0_1_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_salt_pepper_noise(port_input_name='RAW_RESIZE', is_rgb=False, density=0, port_output_name='S&P_NOISE_DENS_0_4_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_gaussian_blur_image_job(port_input_name='RAW_RESIZE', is_rgb=False, sigma=0, port_output_name='GAUSS_BLUR_K_0_S_1_4_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_gaussian_blur_image_job(port_input_name='RAW_RESIZE', is_rgb=False, sigma=0, port_output_name='GAUSS_BLUR_K_0_S_3_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_mean_blur_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel_size=0, port_output_name='MEAN_BLUR_K_9_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_mean_blur_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel_size=0, port_output_name='MEAN_BLUR_K_7_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_motion_blur_filter_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel_size=0, angle=0, port_output_name='MOTION_BLUR_K_7_ANGLE_90_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_motion_blur_filter_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel_size=0, angle=0, port_output_name='MOTION_BLUR_K_7_ANGLE_-90_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, port_output_name='CHANGE_ALPHA_1.5_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, beta=0, port_output_name='CHANGE_BETA_10_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, beta=0, port_output_name='CHANGE_ALPHA_1.5_BETA_10_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, beta=0, port_output_name='CHANGE_ALPHA_2.5_BETA_10_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, beta=0, port_output_name='CHANGE_ALPHA_0.5_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, beta=0, port_output_name='CHANGE_ALPHA_0.1_BETA_50_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_pixelate_image_job(port_input_name='RAW_RESIZE', is_rgb=False, nr_pixels_to_group=3, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_pixelate_image_job(port_input_name='RAW_RESIZE', is_rgb=False, nr_pixels_to_group=2, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_sharpen_filter_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel=None, port_output_name='SHARPEN_laplace_v1_3x3_xy_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_unsharp_filter_expanded_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel=None, port_output_name='UNSHARP_FILTER_laplace_v1_3x3_xy_S_0_5_RAW_RESIZE', strenght=0.5, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_unsharp_filter_expanded_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel=None, port_output_name='UNSHARP_FILTER_laplace_v1_3x3_xy_S_0_8_RAW_RESIZE', strenght=0.8, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    for el in range(len(list_of_ports_to_move.copy())):
        list_of_ports_to_move[el] += '_LC0'

    Application.create_folder_from_list_ports(folder_name='Logs/ml_results/TRAIN_LABEL', list_port=list_of_ports_to_move)

    Utils.close_files()


def feature_examples():
    """
    Create AKAZE feature for each diffusion function on RGB and grey images
    :return:
    """
    Application.set_output_image_folder('Logs/application_results_feature_example')
    Application.set_input_image_folder('TestData/TMBuD/images')
    Application.delete_folder_appl_out()

    raw_img = Application.do_get_image_job(port_output_name='RAW_IMG')
    grey_img = Application.do_grayscale_transform_job(port_input_name='RAW_IMG', port_output_name='GREY_IMG')

    for input in [raw_img, grey_img]:
        for diff_func in [cv2.KAZE_DIFF_WEICKERT, cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2, cv2.KAZE_DIFF_CHARBONNIER]:
            Application.do_kaze_job(port_input_name=input, diffusivity=diff_func)

    Application.create_config_file()
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save='ALL')
    Application.run_application()
    Utils.close_files()


def determining_sharpness():
    """
    Experiment to determine the best sharpenning factor for our usecase
    :return:
    """
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()
    input_source = 'TestData/sharpnnes_test'

    Application.set_output_image_folder('Logs/application_results')
    Application.set_input_image_folder(input_source)

    Application.do_get_image_job(port_output_name='RAW_IMG')
    grey_img = Application.do_grayscale_transform_job(port_input_name='RAW_IMG', port_output_name='GREY_IMG')
    Application.do_pyramid_level_down_job(port_input_name=grey_img, is_rgb=False, port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0, number_of_lvl=2)

    input_list = ([CONFIG.PYRAMID_LEVEL.LEVEL_0, grey_img, '_L0', []],
                  [CONFIG.PYRAMID_LEVEL.LEVEL_1, grey_img, '_L1', []],
                  [CONFIG.PYRAMID_LEVEL.LEVEL_2, grey_img, '_L2', []],
                  )

    for (input_level, input_port, input_app, eval_list) in input_list:
        sharp_list = list()
        sharp_list.append(grey_img)

        sharp_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input_port, is_rgb=False, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=0.7, level=input_level))
        sharp_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input_port, is_rgb=False, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, strenght=0.7, level=input_level))
        sharp_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input_port, is_rgb=False, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strenght=0.7, level=input_level))
        sharp_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input_port, is_rgb=False, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_2, strenght=0.7, level=input_level))
        sharp_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input_port, is_rgb=False, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, strenght=0.7, level=input_level))
        sharp_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input_port, is_rgb=False, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_2, strenght=0.7, level=input_level))

        for el in sharp_list:
            Application.do_histogram_job(port_input_name=el)
            des_output, kp_output, img_output = Application.do_a_kaze_job(port_input_name=el, diffusivity=cv2.KAZE_DIFF_WEICKERT, save_to_npy=False, save_to_text=False, number_features=2048, level=input_level)
            eval_list.append(img_output + input_app)

    Application.create_config_file()
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save='ALL')
    Application.run_application()

    for (input_level, input_port, input_app, eval_list) in input_list:
        Benchmarking.run_SF_benchmark(input_location='Logs/application_results',
                                      raw_image=input_source,
                                      jobs_set=eval_list)

        Benchmarking.run_Entropy_benchmark(input_location='Logs/application_results',
                                           raw_image=input_source,
                                           jobs_set=eval_list)

        for data in ['Entropy', 'SF']:
            Utils.plot_frame_values(name_to_save=data + input_app, eval=eval_list, data=data,
                                    set_name_replace_list=[('A_KAZE_IMG_DT_5_DS_0_THR_0_001_NO_4_NOL_4_D_2_UNSHARP_FILTER_laplace_v1_', 'V1 '),
                                                           ('A_KAZE_IMG_DT_5_DS_0_THR_0_001_NO_4_NOL_4_D_2_UNSHARP_FILTER_laplace_v2_', 'V2 '),
                                                           ('A_KAZE_IMG_DT_5_DS_0_THR_0_001_NO_4_NOL_4_D_2', 'original'),
                                                           ('dilated_', 'Dilated '), ('_xy', ' '), ('_S_', 'S='),
                                                           ('_GREY_IMG_L0', ''), ('_GREY_IMG_L1', ''), ('_GREY_IMG_L2', ''), ('_', '.')],
                                    x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=15, y_ticks_font_size=15, img_size_w=17, img_size_h=10,
                                    legend_name=None, legend_font_size='large', dpi_save_value=1000,
                                    save_plot=True)
    Utils.close_files()

    for (input_level, input_port, input_app, eval_list) in input_list:
        new_port_list = list()
        for el in eval_list:
            new_port_list.append('Number Kp ' + el)

        Utils.plot_custom_list(port_list=new_port_list, set_frame_name=True,
                               set_name_replace_list=[('Number Kp A_KAZE_IMG_DT_5_DS_0_THR_0_001_NO_4_NOL_4_D_2_UNSHARP_FILTER_laplace_v1_', 'V1 '),
                                                      ('Number Kp A_KAZE_IMG_DT_5_DS_0_THR_0_001_NO_4_NOL_4_D_2_UNSHARP_FILTER_laplace_v2_', 'V2 '),
                                                      ('Number Kp A_KAZE_IMG_DT_5_DS_0_THR_0_001_NO_4_NOL_4_D_2', 'original'),
                                                      ('dilated_', 'Dilated '), ('_xy', ' '), ('_S_', 'S='),
                                                      ('_GREY_IMG_L0', ''), ('_GREY_IMG_L1', ''), ('_GREY_IMG_L2', ''), ('_', '.')],
                               name_to_save='Nr_features' + input_app, y_plot_name='Number of features',
                               x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=15, y_ticks_font_size=15, img_size_w=17, img_size_h=10,
                               legend_name=None, legend_font_size='large', dpi_save_value=1000,
                               show_plot=False, save_plot=True)

    Utils.close_files()


def train_model(height, width, n_classes, epochs, steps_per_epoch, val_steps_per_epoch, batch_size, class_names, COLORS, validate_input, validate_gt, class_list_rgb_value):
    """
    Train the model at hand
    :param height: height of the image
    :param width:  width of the image
    :param n_classes: total number of classes
    :param epochs: number of epochs to train
    :param steps_per_epoch: train epoch steps
    :param val_steps_per_epoch: validation epoch step
    :param batch_size: batch size for training
    :param class_names: name of classes for overlay
    :param COLORS: colors for each class to use in overlay
    :param validate_input: location of input data
    :param validate_gt: location of testing data
    :param class_list_rgb_value: correlation between class label and RGB data
    :return: None
    """
    MachineLearning.set_image_input_folder('Logs/ml_results/TRAIN_INPUT')
    MachineLearning.set_label_input_folder('Logs/ml_results/TRAIN_LABEL')
    MachineLearning.set_image_validate_folder('Logs/ml_results/VAL_INPUT')
    MachineLearning.set_label_validate_folder('Logs/ml_results/VAL_LABEL')
    MachineLearning.clear_model_trained()

    MachineLearning.do_semseg_base(model="resnet50_segnet", input_height=height, input_width=width, n_classes=n_classes, epochs=epochs,
                                   verify_dataset=False, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch, optimizer_name='adam', batch_size=batch_size)

    Application.set_input_image_folder(validate_input)
    Application.set_output_image_folder('Logs/application_results_semseg_iou')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')

    Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=n_classes, level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                   save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Benchmarking.run_IoU_benchmark(input_location='Logs/application_results_semseg_iou/',
                                   gt_location=validate_gt,
                                   raw_image=validate_input,
                                   jobs_set=['SEMSEG_RESNET50_SEGNET_RAW_L0'],
                                   class_list_name=class_names, unknown_class=0,
                                   is_rgb_gt=True, show_only_set_mean_value=True,
                                   class_list_rgb_value=class_list_rgb_value)

    Utils.close_files()


def main_bow_create(building_classes, desc, diff, desc_size, nOctaves, nLayes, level, kernel_smoothing, smoothing_strength,
                    thr, thr_akaze, dictionarySize, class_in, class_out, class_names, COLORS, input_file_location, use_gps, gps_file):
    """
    The offline part of the proposed algorithm. All the parameters are passed as list in order to permit multiple runs.
    :param building_classes: number of classes in ROI filtering
    :param desc: descriptors to use
    :param diff: diffusion functions to use
    :param desc_size: descriptor sizes to use
    :param nOctaves: number of octaves to use in AKAZE generation
    :param nLayes: number of layers to use in AKAZE generation
    :param thr_akaze: threshold values for AKAZE generation
    :param level: pyramid levels to use
    :param kernel_smoothing: kernel to use for UM filtering
    :param smoothing_strength: kernel smoothing strengths to use for UM filtering
    :param dictionarySize: dictionary cluster sizes to use
    :param thr: distance threshold to use for BOF FLANN search sizes to use
    :param class_in: list of input classes to use for correlation
    :param class_out: list of output of classes to use for correlation
    :param class_names: list of class names for output class labels
    :param COLORS: list of colors for output class labels
    :param input_file_location: location of input images
    :param use_gps: if we want to use GPS tags in creating the BOF
    :param gps_file: location of GPS file
    :return: None
    """
    Application.set_input_image_folder(input_file_location)
    Application.set_output_image_folder('Logs/application_results')
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW', direct_grey=False)
    Application.do_pyramid_level_down_job(port_input_name='RAW', number_of_lvl=int(level[-1]), port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0, is_rgb=True)

    grey = Application.do_grayscale_transform_job(port_input_name='RAW', level=level, port_output_name='GRAW')
    Application.do_get_data_TMBuD_csv_job(csv_field='Coordinates Landmark', file_location=gps_file, port_img_output='GPS_LANDMARK', level=level)

    grey_smooth = Application.do_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, kernel=kernel_smoothing, strenght=smoothing_strength,
                                                             level=level, port_output_name='UM_GRAW_' + kernel_smoothing[-1] + '_' + str(smoothing_strength)[-1])

    semseg_image = Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=len(class_names), level=level, port_name_output='SEMSEG_RAW',
                                                  save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)

    binary_mask = Application.do_class_correlation(port_input_name=semseg_image, class_list_in=class_in, class_list_out=class_out, level=level, port_output_name='ROI_RAW')

    kp, des, img = Application.do_a_kaze_job(port_input_name=grey_smooth, descriptor_channels=1, mask_port_name=binary_mask,
                                             descriptor_size=desc_size, descriptor_type=desc, diffusivity=diff, save_to_text=False,
                                             threshold=thr_akaze, nr_octaves=nOctaves, nr_octave_layers=nLayes, level=level)

    Application.do_tmbud_bow_job(port_to_add=des, dictionary_size=dictionarySize, use_gps=use_gps, gps_port='GPS_LANDMARK',
                                 number_classes=building_classes, level=level)

    Application.create_config_file()
    # Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save='ALL')
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save=[])
    Application.run_application()

    Utils.close_files()


def main_bow_inquiry(building_classes, desc, diff, desc_size, nOctaves, nLayes, thr, input_file_location, distance, use_gps,
                     thr_akaze, dictionarySize, class_in, class_out, class_names, COLORS, level, kernel_smoothing, smoothing_strength,
                     threshold_matching, csv_landmark, gt_location, gps_file, boxing, tracking, roi):
    """
    The online part of the proposed algorithm. All the parameters are passed as list in order to permit multiple runs.
    :param building_classes: number of classes in ROI filtering
    :param desc: list of descriptors to use
    :param diff: list of diffusion functions to use
    :param desc_size: list of descriptor sizes to use
    :param nOctaves: list of number of octaves to use in AKAZE generation
    :param nLayes: list of number of layers to use in AKAZE generation
    :param thr_akaze: list of threshold values for AKAZE generation
    :param level: pyramid levels to use
    :param kernel_smoothing: list of kernel to use for UM filtering
    :param smoothing_strength: list of kernel smoothing strengths to use for UM filtering
    :param dictionarySize: list of dictionary cluster sizes to use
    :param thr: list of distance threshold to use for BOF FLANN search sizes to use
    :param class_in: list of input classes to use for correlation
    :param class_out: list of output of classes to use for correlation
    :param class_names: list of class names for output class labels
    :param COLORS: list of colors for output class labels
    :param input_file_location: location of input images
    :param use_gps: if we want to use GPS tags in creating the BOF
    :param gps_file: location of GPS file
    :param boxing: if we want to add a box on detection
    :param roi: if we want to add a semseg overlay
    :param tracking: if we want to use tracking
    :param threshold_matching: minimum percent of features matched to be taken in consideration
    :param csv_landmark: csv file location for landmarks name - object class
    :param gt_location: location of ground truth location
    :return: None
    """
    list_to_eval = list()

    Application.set_input_image_folder(input_file_location)
    Application.set_output_image_folder('Logs/query_application')
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW', direct_grey=False)
    Application.do_pyramid_level_down_job(port_input_name='RAW', number_of_lvl=int(level[-1]), port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0, is_rgb=True)

    grey = Application.do_grayscale_transform_job(port_input_name='RAW', level=level, port_output_name='GRAW')
    Application.do_get_data_TMBuD_csv_job(csv_field='Coordinates image', file_location=gps_file, port_img_output='GPS_LANDMARK',
                                          level=level)

    grey_smooth = Application.do_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, kernel=kernel_smoothing, strenght=smoothing_strength,
                                                             level=level, port_output_name='UM_GRAW_' + kernel_smoothing[-1] + '_' + str(smoothing_strength)[-1])

    semseg_image = Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=3, level=level, port_name_output='SEMSEG_RAW',
                                                  save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)

    binary_mask = Application.do_class_correlation(port_input_name=semseg_image, class_list_in=class_in, class_list_out=class_out, level=level, port_output_name='ROI_RAW')

    kp, des, img = Application.do_a_kaze_job(port_input_name=grey_smooth, descriptor_channels=1, mask_port_name=binary_mask,
                                             descriptor_size=desc_size, descriptor_type=desc, diffusivity=diff,  save_to_text=False,
                                             threshold=thr_akaze, nr_octaves=nOctaves, nr_octave_layers=nLayes, level=level)

    if roi:
        mask = binary_mask
    else:
        mask = None

    final = Application.do_tmbud_bow_inquiry_flann_job(port_to_inquiry_des=des, port_to_inquiry_kp=kp, port_to_inquire_img=grey, level=level,
                                                       saved_to_npy=True, saved_to_text=False, number_classes=building_classes, save_img_detection=True,
                                                       flann_thr=thr, threshold_matching=threshold_matching, name_landmark_port=csv_landmark,
                                                       mask_port=mask, if_tracking=tracking, if_box_on_detection=boxing,
                                                       location_of_bow='Logs/application_results', use_gps=use_gps, gps_port='GPS_LANDMARK', distante_accepted=distance,
                                                       bow_port='ZuBuD_BOW_' + dictionarySize.__str__() + '_' + des + '_' + level,
                                                       port_out_image='FINAL')

    list_to_eval.append(final + '_' + level)

    Application.create_config_file()
    # Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save='ALL')
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save=[])
    Application.configure_show_pictures(ports_to_show=['FINAL_L2'], time_to_show=100)
    Application.run_application()

    Benchmarking.run_CBIR_ZuBuD_benchmark(input_location='Logs/query_application/',
                                          gt_location=gt_location,
                                          raw_image=input_file_location,
                                          jobs_set=list_to_eval)

    Utils.close_files()


def main_bow_inquiry_movie(building_classes, desc, diff, desc_size, nOctaves, nLayes, thr, input_file_location, distance, use_gps,
                     thr_akaze, dictionarySize, class_in, class_out, class_names, COLORS, level, kernel_smoothing, smoothing_strength,
                     threshold_matching, csv_landmark, gt_location, gps_file, boxing, tracking, roi):
    """
    The online part of the proposed algorithm. All the parameters are passed as list in order to permit multiple runs.
    :param building_classes: number of classes in ROI filtering
    :param desc: list of descriptors to use
    :param diff: list of diffusion functions to use
    :param desc_size: list of descriptor sizes to use
    :param nOctaves: list of number of octaves to use in AKAZE generation
    :param nLayes: list of number of layers to use in AKAZE generation
    :param thr_akaze: list of threshold values for AKAZE generation
    :param level: pyramid levels to use
    :param kernel_smoothing: list of kernel to use for UM filtering
    :param smoothing_strength: list of kernel smoothing strengths to use for UM filtering
    :param dictionarySize: list of dictionary cluster sizes to use
    :param thr: list of distance threshold to use for BOF FLANN search sizes to use
    :param class_in: list of input classes to use for correlation
    :param class_out: list of output of classes to use for correlation
    :param class_names: list of class names for output class labels
    :param COLORS: list of colors for output class labels
    :param input_file_location: location of input images
    :param use_gps: if we want to use GPS tags in creating the BOF
    :param gps_file: location of GPS file
    :param boxing: if we want to add a box on detection
    :param roi: if we want to add a semseg overlay
    :param tracking: if we want to use tracking
    :param threshold_matching: minimum percent of features matched to be taken in consideration
    :param csv_landmark: csv file location for landmarks name - object class
    :param gt_location: location of ground truth location
    :return: None
    """
    list_to_eval = list()

    Application.set_input_image_folder(input_file_location)
    Application.set_output_image_folder('Logs/query_application')
    Application.delete_folder_appl_out()

    # Application.set_input_video(r'c:\repos\CM_dataset\movies\20201104_155714.mp4')
    # file_csv = r'c:\repos\CM_dataset\movies\20201104_155714.csv'
    # distance = 70
    # threshold_matching = 1.5

    Application.set_input_video(r'c:\repos\CM_dataset\movies\20201229_154044.mp4')
    file_csv = r'c:\repos\CM_dataset\movies\20201229_154044.csv'
    distance = 65
    # for L1
    threshold_matching = 1.5
    # for L2
    # threshold_matching = 1.5

    Application.do_get_video_job(port_output_name='RAW', rotate_image=False, name_of_frame='qimg{0:04d}.png')
    Application.do_pyramid_level_down_job(port_input_name='RAW', number_of_lvl=int(level[-1]), port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0, is_rgb=True)
    grey = Application.do_grayscale_transform_job(port_input_name='RAW', level=level, port_output_name='GRAW')
    Application.do_get_data_TMBuD_csv_job(csv_field='Coordinates image', file_location=file_csv, port_img_output='GPS_LANDMARK', level=level)

    grey_smooth = Application.do_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, kernel=kernel_smoothing, strenght=smoothing_strength,
                                                             level=level, port_output_name='UM_GRAW_' + kernel_smoothing[-1] + '_' + str(smoothing_strength)[-1])

    semseg_image = Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=3, level=level,
                                                  save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS,
                                                  port_name_output='SEMSEG_RAW')

    binary_mask = Application.do_class_correlation(port_input_name=semseg_image, class_list_in=class_in, class_list_out=class_out, level=level, port_output_name='ROI_RAW')

    kp, des, img = Application.do_a_kaze_job(port_input_name=grey_smooth, descriptor_channels=1, mask_port_name=binary_mask,
                                             descriptor_size=desc_size, descriptor_type=desc, diffusivity=diff,  save_to_text=False,
                                             threshold=thr_akaze, nr_octaves=nOctaves, nr_octave_layers=nLayes, level=level)
    if roi:
        mask = binary_mask
    else:
        mask = None

    final = Application.do_tmbud_bow_inquiry_flann_job(port_to_inquiry_des=des, port_to_inquiry_kp=kp, port_to_inquire_img=grey, level=level,
                                                       saved_to_npy=True, saved_to_text=False, number_classes=building_classes, save_img_detection=True,
                                                       flann_thr=thr, threshold_matching=threshold_matching, name_landmark_port=csv_landmark,
                                                       mask_port=mask, if_tracking=tracking, if_box_on_detection=boxing,
                                                       location_of_bow='Logs/application_results', use_gps=use_gps, gps_port='GPS_LANDMARK', distante_accepted=distance,
                                                       bow_port='ZuBuD_BOW_' + dictionarySize.__str__() + '_' + des + '_' + level,
                                                       port_out_image='FINAL')

    Application.create_config_file()

    ports_to_save = ['RAW_' + level, 'OVERLAY_SEMSEG_RAW_' + level, img + '_' + level, 'FINAL_' + level]
    ports_to_show = ['FINAL_' + level]

    # Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save=ports_to_save)
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save='ALL')
    Application.configure_show_pictures(ports_to_show=ports_to_show, time_to_show=1)

    Application.run_application()
    Utils.close_files()

    name = (file_csv.split('.')[0]).split('\\')[-1]
    print(name)
    Utils.create_video(port_list=ports_to_save, folder_of_ports='Logs/query_application', fps=15, name='{}_details.mp4'.format(name))
    Utils.create_video(port_list=ports_to_show, folder_of_ports='Logs/query_application', fps=25, name='{}_output.mp4'.format(name))


if __name__ == "__main__":
    # feature detection examples
    # feature_examples()

    # determine best sharpening parameters
    # determining_sharpness()

    w = 320
    h = 512

    #            [BACKGROUND,  BUILDING,      NOISE]
    BDT_COLORS = [(0, 0, 0), (125, 125, 0), (0, 0, 255)]
    BDT_CLASSES = [0,           1,              2]

    dataset_processed = 'Logs/bulk_data'
    dataset_input_img = 'Logs/bulk_data/img/RAW_L0'
    dataset_input_labels_processed = 'Logs/bulk_data/labels/BDT_LABELS_L0'

    # dataset_input_labels = r'c:\repos\eecvf_git\TestData\building_labels_database\eTRIMS\annotations\08_etrims-ds'
    # dataset_input_img = r'c:\repos\eecvf\TestData\building_labels_database\eTRIMS\images\image'
    # #                          VARIOUS      BUILDING        CAR            DOOR          PAVEMENT         ROAD           SKY       VEGETATION      WINDOW
    # LabelMe_COLORS =          [(0, 0, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (0, 64, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0)]
    # LabelMe_BDT_CORRELATION = [     2,          1,           2,             1,           2,                 2,           2,              2,          1]
    # prepare_dataset_img(dataset_img_input=dataset_input_img, dataset_out=dataset_processed + '/img')
    # Utils.reopen_files()
    # prepare_dataset_labels(dataset_in=dataset_input_labels, dataset_out=dataset_processed + '/labels', LabelMe_COLORS=LabelMe_COLORS, LabelMe_BDT_CORRELATION=LabelMe_BDT_CORRELATION, BDT_COLORS=BDT_COLORS, BDT_CLASSES=BDT_CLASSES)
    # Utils.reopen_files()

    # dataset_input_labels = r'c:\repos\eecvf_git\TestData\building_labels_database\LabelMeFacade\labels'
    # dataset_input_img = r'c:\repos\eecvf\TestData\building_labels_database\LabelMeFacade\images'
    # #                          VARIOUS      BUILDING        CAR            DOOR          PAVEMENT         ROAD           SKY       VEGETATION      WINDOW
    # eTRIMS_COLORS =          [(0, 0, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (0, 64, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0)]
    # eTRIMS_BDT_CORRELATION = [     2,          1,           2,             1,           2,                 2,           2,              2,          1]
    # prepare_dataset_img(dataset_img_input=dataset_input_img, dataset_out=dataset_processed + '/img')
    # Utils.reopen_files()
    # prepare_dataset_labels(dataset_in=dataset_input_labels, dataset_out=dataset_processed + '/labels', LabelMe_COLORS=eTRIMS_COLORS, LabelMe_BDT_CORRELATION=eTRIMS_BDT_CORRELATION, BDT_COLORS=BDT_COLORS, BDT_CLASSES=BDT_CLASSES )
    # Utils.reopen_files()

    # dataset_input_labels = r'c:\repos\eecvf_git\TestData\TMBuD\parsed_dataset\SEMSEG_EVAL_FULL\label_full\TRAIN\png'
    # dataset_input_img = r'c:\repos\eecvf_git\TestData\TMBuD\parsed_dataset\SEMSEG_EVAL_FULL\img_label_full\TRAIN\png'
    # # class_names = [           "UNKNOWN", "BUILDING",      "DOOR",         "WINDOW",      "SKY",   "VEGETATION",   "GROUND",       "NOISE"]
    # TMBuD_COLORS =       [(     0, 0, 0), (125, 125, 0), (0, 125, 125), (0, 255, 255), (255, 0, 0), (0, 255, 0), (125, 125, 125), (0, 0, 255)]
    # TMBuD_BDT_CORRELATION = [     0,          1,                1,             1,           2,           2,           2,              2]
    # prepare_dataset_img(dataset_img_input=dataset_input_img, dataset_out=dataset_processed + '/img')
    # Utils.reopen_files()
    # prepare_dataset_labels(dataset_in=dataset_input_labels, dataset_out=dataset_processed + '/labels', LabelMe_COLORS=TMBuD_COLORS, LabelMe_BDT_CORRELATION=TMBuD_BDT_CORRELATION, BDT_COLORS=BDT_COLORS, BDT_CLASSES=BDT_CLASSES)
    # Utils.reopen_files()

    # main_training_data(width=w, height=h, data_input_img=dataset_input_img)
    # Utils.reopen_files()
    # main_training_label(width=w, height=h, dataset_input=dataset_input_labels_processed)
    # Utils.reopen_files()

    # class_names = ["UNKNOWN", "BUILDING", "DOOR", "WINDOW", "SKY", "VEGETATION", "GROUND", "NOISE"]
    # COLORS_TMBuD = [(     0, 0, 0), (125, 125, 0), (0, 125, 125), (0, 255, 255), (255, 0, 0), (0, 255, 0), (125, 125, 125), (0, 0, 255)]
    # TMBuD_CORRELATION = [     0,          1,           1,             1,           2,                 2,           2,              2]
    # prepare_dataset_TMBuD(COLORS_TMBuD=COLORS_TMBuD, TMBuD_CORRELATION=TMBuD_CORRELATION)
    # Utils.reopen_files()

    n_classes = 3
    epochs = 350
    batch_size = 6
    train_nr_images = len(os.listdir(r'c:\repos\eecvf_git\Logs\ml_results\TRAIN_INPUT'))
    val_nr_images = len(os.listdir(r'c:\repos\eecvf_git\Logs\ml_results\VAL_INPUT'))
    steps_per_epoch = int((train_nr_images/epochs)/batch_size)
    val_steps_per_epoch = int(val_nr_images/batch_size)
    class_names = ["UNKNOWN", "BUILDING", "NOISE"]
    COLORS = [(0, 0, 0), (125, 125, 0), (0, 0, 255)]
    validate_input = 'TestData/TMBuD/parsed_dataset/SEMSEG_EVAL_FULL/img_label_full/TEST/png'
    validate_gt = 'Logs/TMBuD/labels/BDT_LABELS_PNG_L0'
    class_list_rgb_value = [0, 87, 76]

    # train_model(width=w, height=h, n_classes=n_classes, epochs=epochs, val_steps_per_epoch=val_steps_per_epoch, batch_size=batch_size, steps_per_epoch=steps_per_epoch, class_names=class_names, COLORS=COLORS,
    #             validate_input=validate_input, validate_gt=validate_gt, class_list_rgb_value=class_list_rgb_value, )
    # Utils.reopen_files()


    # TRAIN_input_file = r'c:\repos\Building detection\ZuBud_dataset\png_ZuBuD_parsed'
    # TEST_input_file = r'c:\repos\Building detection\ZuBud_dataset\qimage'
    # gt_location = r"c:\repos\Building detection\ZuBud_dataset\zubud_groundtruth.txt"
    # nr_classes = 201

    # TRAIN_input_file = r'c:\repos\Building detection\ZuBud_dataset\png_ZuBuD_parsed'
    # TEST_input_file = r'c:\repos\Building detection\ZuBuD+\test_balanced'
    # gt_location = r"c:\repos\Building detection\ZuBuD+\ground_truth_balanced.txt"
    # nr_classes = 201

    TRAIN_input_file = 'TestData/TMBuD/parsed_dataset/v3_2/TRAIN'
    TEST_input_file = 'TestData/TMBuD/parsed_dataset/v3_2/TEST'
    csv_landmark = 'TestData/TMBuD/parsed_dataset/v3_2/landmarks.csv'
    gt_location = 'TestData/TMBuD/parsed_dataset/v3_2/TMBuD_groundtruth.txt'
    train_csv_file = 'TestData/TMBuD/parsed_dataset/v3_2/train_data.csv'
    test_csv_file = 'TestData/TMBuD/parsed_dataset/v3_2/test_data.csv'
    nr_classes = 125

    # TRAIN_input_file = 'TestData/TMBuD/parsed_dataset/v3_2_night/TRAIN'
    # TEST_input_file = 'TestData/TMBuD/parsed_dataset/v3_2_night/TEST'
    # csv_landmark = 'TestData/TMBuD/parsed_dataset/v3_2_night/landmarks.csv'
    # gt_location = 'TestData/TMBuD/parsed_dataset/v3_2_night/TMBuD_groundtruth.txt'
    # train_csv_file = 'TestData/TMBuD/parsed_dataset/v3_2_night/train_data.csv'
    # test_csv_file = 'TestData/TMBuD/parsed_dataset/v3_2_night/test_data.csv'
    # nr_classes = 56

    # TRAIN_input_file = 'TestData/TMBuD/parsed_dataset/v3_n/TRAIN'
    # TEST_input_file = 'TestData/TMBuD/parsed_dataset/v3_n/TEST'
    # csv_landmark = 'TestData/TMBuD/parsed_dataset/v3_n/landmarks.csv'
    # gt_location = 'TestData/TMBuD/parsed_dataset/v3_n/TMBuD_groundtruth.txt'
    # train_csv_file = 'TestData/TMBuD/parsed_dataset/v3_n/train_data.csv'
    # test_csv_file = 'TestData/TMBuD/parsed_dataset/v3_n/test_data.csv'
    # nr_classes = 125

    # desc_list = [cv2.AKAZE_DESCRIPTOR_KAZE, cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT, cv2.AKAZE_DESCRIPTOR_MLDB, cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT]
    # diff_list = [cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2, cv2.KAZE_DIFF_CHARBONNIER, cv2.KAZE_DIFF_WEICKERT]

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', sys.argv)
    config = sys.argv[1]
    desc_list = int(sys.argv[2])
    diff_list = int(sys.argv[3])
    desc_size_list = int(sys.argv[4])
    nOctaves_list = int(sys.argv[5])
    nLayes_list = int(sys.argv[6])
    thr_list = float(sys.argv[7])
    thr_akaze_list = float(sys.argv[8])
    dictionarySize_list = int(sys.argv[9])
    kernel_smoothing = sys.argv[10]
    smoothing_strength = float(sys.argv[11])
    use_gps = bool(sys.argv[12])
    distance_list = float(sys.argv[13])

    # desc_list = cv2.AKAZE_DESCRIPTOR_KAZE
    # diff_list = cv2.KAZE_DIFF_PM_G1
    # desc_size_list = 8
    # nOctaves_list = 6
    # nLayes_list = 3
    # thr_akaze_list = 0.001
    # thr_list = 0.8
    # dictionarySize_list = 300
    # kernel_smoothing = CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1
    # smoothing_strength = 0.9
    # use_gps = True
    # distance_list = 100

    pyramid_level = CONFIG.PYRAMID_LEVEL.LEVEL_2
    class_in = [0, 1, 2]
    class_out = [0, 1, 0]

    thr_match_procent = 1
    tracking = True
    boxing = False
    overlay_semseg = True

    Utils.reopen_files()

    if config == 'create_bow':
    # if True:
        main_bow_create(building_classes=nr_classes, desc=desc_list, diff=diff_list, desc_size=desc_size_list,
                        level=pyramid_level, kernel_smoothing=kernel_smoothing, smoothing_strength=smoothing_strength,
                        nOctaves=nOctaves_list, nLayes=nLayes_list, thr=thr_list, thr_akaze=thr_akaze_list, dictionarySize=dictionarySize_list,
                        class_in=class_in, class_out=class_out, class_names=class_names, COLORS=COLORS, input_file_location=TRAIN_input_file, use_gps=use_gps, gps_file=train_csv_file)
        Utils.reopen_files()

    # if config == 'inquiry':
    # if True:
    #     main_bow_inquiry(building_classes=nr_classes, desc=desc_list, diff=diff_list, desc_size=desc_size_list,
    #                      level=pyramid_level, kernel_smoothing=kernel_smoothing, smoothing_strength=smoothing_strength,
    #                      nOctaves=nOctaves_list, nLayes=nLayes_list, thr=thr_list, thr_akaze=thr_akaze_list, dictionarySize=dictionarySize_list,
    #                      class_in=class_in, class_out=class_out, class_names=class_names, COLORS=COLORS, gt_location=gt_location, tracking=tracking, boxing=boxing, roi=overlay_semseg,
    #                      input_file_location=TEST_input_file, distance=distance_list, use_gps=use_gps, threshold_matching=thr_match_procent, csv_landmark=csv_landmark, gps_file=test_csv_file)
    #     Utils.reopen_files()

    if config == 'inquiry':
    # if True:
        main_bow_inquiry_movie(building_classes=nr_classes, desc=desc_list, diff=diff_list, desc_size=desc_size_list,
                               level=pyramid_level, kernel_smoothing=kernel_smoothing, smoothing_strength=smoothing_strength,
                               nOctaves=nOctaves_list, nLayes=nLayes_list, thr=thr_list, thr_akaze=thr_akaze_list, dictionarySize=dictionarySize_list,
                               class_in=class_in, class_out=class_out, class_names=class_names, COLORS=COLORS, gt_location=gt_location, tracking=tracking, boxing=boxing, roi=overlay_semseg,
                               input_file_location=TEST_input_file, distance=distance_list, use_gps=use_gps, threshold_matching=thr_match_procent, csv_landmark=csv_landmark, gps_file=test_csv_file)
        Utils.reopen_files()
