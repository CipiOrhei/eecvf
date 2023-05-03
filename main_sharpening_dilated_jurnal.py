import os.path
import numpy as np
import cv2

import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils

"""
This module contains the code used for the following paper:

"""

def test_single_line_effect():
        """

        """
        Application.delete_folder_appl_out()
        Benchmarking.delete_folder_benchmark_out()

        Application.set_input_image_folder(r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_blur")
        # Application.set_input_image_folder(r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_original")
        # Application.set_input_image_folder('TestData/smoke_test')
        raw = Application.do_get_image_job('RAW')
        grey = Application.do_grayscale_transform_job(port_input_name='RAW')

        eval_list = list()

        eval_list.append(grey)

        for (input, is_rgb) in ([grey, False],):
            # Classical sharpening with HPF
            eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1))
            eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1))
            eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1))
            eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1))
            eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1))
            eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1))
            eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1))

            # Classical UM
            strength = 0.1
            eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=strength))
            eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, strenght=strength))
            eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, strenght=strength))
            eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, strenght=strength))
            eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strenght=strength))
            eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, strenght=strength))
            eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, strenght=strength))

            # Nonlinear UM
            gk = 0.75; hpf_strength = 0.01
            eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk, operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, hpf_strength=hpf_strength))
            eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk, operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, hpf_strength=hpf_strength))
            eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk, operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, hpf_strength=hpf_strength))
            eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk, operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, hpf_strength=hpf_strength))
            eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk, operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, hpf_strength=hpf_strength))
            eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk, operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, hpf_strength=hpf_strength))
            eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk, operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, hpf_strength=hpf_strength))

            # Normalized NUM
            strength = 0.15
            eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, strenght=strength))
            eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=strength))
            eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, strenght=strength))
            eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, strenght=strength))
            eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, strenght=strength))
            eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strenght=strength))
            eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, strenght=strength))
            eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, strenght=strength))

            # ANUM
            bf_d = 19;  bf_s = 30; bf_c = 30; t1 = 5; t2 = 60
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, thr_1=t1, thr_2=t2, bf_distance=bf_d,bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, thr_1=t1, thr_2=t2, bf_distance=bf_d,bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, thr_1=t1, thr_2=t2, bf_distance=bf_d,bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, thr_1=t1, thr_2=t2, bf_distance=bf_d,bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, thr_1=t1, thr_2=t2, bf_distance=bf_d,bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, thr_1=t1, thr_2=t2, bf_distance=bf_d,bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, thr_1=t1, thr_2=t2, bf_distance=bf_d,bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, thr_1=t1, thr_2=t2, bf_distance=bf_d,bf_sigma_space=bf_s, bf_sigma_colors=bf_c))

            # AUM
            t1 = 60; t2 = 200; dl = 3; dh = 4; mu = 0.2; beta = 0.5
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, thr_1=t1, thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, thr_1=t1, thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, thr_1=t1, thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, thr_1=t1, thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, thr_1=t1, thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, thr_1=t1, thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, thr_1=t1, thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, thr_1=t1, thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))

            # CUM
            hpf_strength = 0.7; win_size = 7; sigma = 35; thr = 5
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))

            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))

            # Selective UM
            t1 = 0.3; t2 = 0.8; bf_d = 19; ss = 40; T = 5
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, lambda_v=t1, lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=ss, T=T))
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, lambda_v=t1, lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=ss, T=T))
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, lambda_v=t1, lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=ss, T=T))
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, lambda_v=t1, lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=ss, T=T))
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, lambda_v=t1, lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=ss, T=T))
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, lambda_v=t1, lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=ss, T=T))
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, lambda_v=t1, lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=ss, T=T))
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, lambda_v=t1, lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=ss, T=T))

            # Histogram Equalization UM
            str_1 = 0.8; str_2 = 0.05
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, strength_1=str_1, strength_2=str_2))
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strength_1=str_1, strength_2=str_2))
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, strength_1=str_1, strength_2=str_2))
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, strength_1=str_1, strength_2=str_2))
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, strength_1=str_1, strength_2=str_2))
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strength_1=str_1, strength_2=str_2))
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, strength_1=str_1, strength_2=str_2))
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, strength_1=str_1, strength_2=str_2))

        for el in eval_list:
            Application.do_histogram_job(port_input_name=el)
            Application.do_plot_lines_over_columns_job(port_input_name=el, lines=[50], columns=[240, 280])

        Application.create_config_file()
        Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
        # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
        Application.run_application()

        for el in range(len(eval_list)):
            eval_list[el] += '_L0'

        Benchmarking.run_PSNR_benchmark(input_location='Logs/application_results',
                                       gt_location=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_original",
                                       raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_original",
                                       jobs_set=eval_list, db_calc=False)

        Utils.close_files()

        suit_list = ['HPF', 'N_NUM', 'NUM', 'ANUM', 'AUM', 'C_CUM', 'CUM','SUM', 'HE_UM', 'UM']

        for el in suit_list:
            sub_list_to_plot = ['GRAY_RAW_L0'] + [string for string in eval_list if string.startswith(el)]
            plot_multiple_lines_image(list_ports=sub_list_to_plot, plot_name=el, lines=[50], columns=[240, 270], name=el, img_name='test')


def plot_ratio_PSNR_runtime():
    import matplotlib.pyplot as plt
    # Data
    psnr_data = {
        "HPF": [33.161, 32.406, 33.353, 29.524, 32.848, 28.224, 31.400],
        "UM": [33.059, 33.141, 33.080, 32.971, 33.170, 29.362, 33.338],
        "NUM": [33.221, 34.071, 33.843, 30.654, 34.305, 29.350, 33.857],
        "N_NUM": [32.899, 32.925, 32.989, 33.029, 33.071, 33.075, 33.223],
        "ANUM": [34.034, 32.986, 34.226, 24.377, 33.603, 17.548, 31.752],
        "AUM": [33.105, 33.070, 33.435, 29.942, 33.349, 28.507, 32.422],
        "CUM": [33.516, 34.548, 34.062, 39.222, 34.742, 40.101, 35.817],
        "C_CUM": [33.956, 35.864, 34.460, 46.596, 35.237, 62.213, 36.220],
        "SUM": [33.058, 33.317, 33.181, 30.699, 33.483, 28.344, 33.633],
        'HE_UM' : [11.733, 11.655, 11.667, 11.531, 11.782, 11.834, 11.725],
    }

    runtime_data = {
        "HPF": [1.996, 2.646, 1.644, 5.024, 1.713, 6.032, 1.824],
        "UM": [2.231, 2.786, 2.255, 3.039, 2.180, 3.878, 2.598],
        "NUM": [7.139, 7.323, 6.471, 7.835, 6.881, 7.991, 6.610],
        "N_NUM": [4.604, 5.082, 4.656, 6.113, 4.565, 6.571, 5.628],
        "ANUM": [265.655, 271.236, 264.191, 266.394, 267.206, 268.633, 265.769],
        "AUM": [304.833, 305.492, 303.915, 304.706, 304.903, 305.984, 306.411],
        "CUM": [132.085, 131.323, 131.701, 132.468, 132.406, 133.021, 132.313],
        "C_CUM": [133.101, 132.351, 133.169, 135.008, 133.061, 137.315, 135.814],
        "SUM": [202.530, 204.847, 204.151, 210.971, 203.052, 207.853, 209.866],
        "HE_UM": [54.018, 54.068, 54.174, 55.010, 53.769, 56.280, 54.587],
    }

    kernel_sizes = ['3', '5',  '5(d)', '7', '7(d)', '9', '9(d)']

    # Calculate PSNR gain and runtime relative increase
    for method, values in psnr_data.items():
        psnr_gain = np.array(psnr_data[method]) - psnr_data["HPF"][0]  # Subtract the PSNR value of the 3x3 kernel
        # runtime_relative_increase = np.array(runtime_data[method]) / runtime_data["HPF"][0]  # Divide by the runtime value of the 3x3 kernel
        runtime_relative_increase = np.array(runtime_data[method]) / runtime_data["HPF"][0]  # Divide by the runtime value of the 3x3 kernel
        ratio = psnr_gain / runtime_relative_increase
        plt.scatter(kernel_sizes, ratio, marker='o', label=method)

    # Plot settings
    fig = plt.gcf()
    fig.set_size_inches(w=15, h=10)
    plt.xlabel('Kernel Size',  fontsize=25)
    plt.ylabel('PSNR Gain/Runtime Relative Increase',  fontsize=25)
    # plt.title('Trade-off between PSNR Gain and Runtime Relative Increase relative to HPF 3x3', fontsize=22)
    plt.legend()
    plt.grid()
    # plt.show()
    file_to_save = os.path.join(CONFIG.APPL_SAVE_LOCATION)
    if not os.path.exists(file_to_save):
        os.makedirs(file_to_save)

    plt.savefig(os.path.join(file_to_save, '{}.png'.format('syntethic_line_psnr_vs_runtime')), bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_multiple_lines_image(list_ports, img_name, lines, columns, name, plot_name):
    import re
    pixel_line_series = list()
    pixel_column_series = list()
    data_series_names = list()

    pixel_line_series.append([50]*(255-columns[0]) + [200]*(columns[1]-255))
    pixel_column_series.append(np.arange(columns[0], columns[1], 1))
    data_series_names.append('ideal')

    for port in list_ports:
        location = os.path.join(CONFIG.APPL_SAVE_LOCATION, port, port + '_' + img_name  + CONFIG.APPl_SAVE_PICT_EXTENSION)
        img = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
        for i,j in zip([r'.*(3x3).*', r'.*(dilated_5x5).*', r'.*(dilated_7x7).*', r'.*(dilated_9x9).*', r'.*(5x5).*', r'.*(7x7).*', r'.*(9x9).*', r'.*(GRAY_RAW).*'],
                       ['3x3','5x5(d)','7x7(d)','9x9(d)','5x5','7x7','9x9', 'original']):
            name = re.sub(i, j, port)
            if name != port:
                data_series_names.append(name)
                break

        for line in lines:
            if len(columns) == 0:
                pixel_line_series.append(np.array(img[line, :]))
                pixel_column_series.append(np.arange(0, len(img[0]), 1))
            else:
                pixel_line_series.append(np.array(img[line, columns[0]:columns[1]]))
                pixel_column_series.append(np.arange(columns[0], columns[1], 1))

    Utils.plotting.plot_custom_series_list(data_series=pixel_line_series, data_axis=pixel_column_series, x_plot_name='Line pixel',
                                y_plot_name='Pixel Value',
                                y_min=0, y_max=260,
                                title_name=plot_name, title_font_size=22,
                                data_series_names=data_series_names, legend_name=None, show_legend=True, if_grid=True,
                                name_folder='PLOT_LINE_suits',
                                name_to_save='plot_line_{}'.format(plot_name),
                                save_plot=True, show_plot=False)


if __name__ == "__main__":
    test_single_line_effect()
    plot_ratio_PSNR_runtime()
