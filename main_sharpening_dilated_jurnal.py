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
  title={An analysis of extended and dilated filters in sharpening algorithms},
  author={Orhei, Ciprian and Vasiu, Radu},
  booktitle={--},
  pages={--},
  year={2023},
  organization={IEEE}

"""

def test_single_line_effect():
        """

        """
        # Application.delete_folder_appl_out()
        # Benchmarking.delete_folder_benchmark_out()

        Application.set_input_image_folder(r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_blur")
        # Application.set_input_image_folder(r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_original")
        # Application.set_input_image_folder('TestData/smoke_test')
        raw = Application.do_get_image_job('RAW')
        grey = Application.do_grayscale_transform_job(port_input_name=raw)

        eval_list = list()
        eval_list.append(grey)

        for (input, is_rgb) in ([grey, False],):
            for kernel in [
                CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,
                CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1,
                CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1,  CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1
            ]:
                # Classical sharpening with HPF
                eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel))
                # Classical UM
                strength = 0.1
                eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength))
                # Nonlinear UM
                gk = 0.4; hpf_strength = 0.025
                eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk,
                                                                         operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=kernel,
                                                                         hpf_strength=hpf_strength))
                # Normalized NUM
                strength = 0.125
                eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength))
                # ANUM
                bf_d = 19;  bf_s = 30; bf_c = 30; t1 = 5; t2 = 60
                eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                                                                                       thr_1=t1, thr_2=t2, bf_distance=bf_d,
                                                                                       bf_sigma_space=bf_s, bf_sigma_colors=bf_c))

                # AUM
                t1 = 60; t2 = 200; dl = 3; dh = 4; mu = 0.2; beta = 0.5
                eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, thr_1=t1,
                                                                            thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
                # CUM
                hpf_strength = 0.7; win_size = 7; sigma = 35; thr = 5
                eval_list.append(
                    Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                                                                           hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma,
                                                                           lee_sigma_filter_window=win_size, threshold_cliping_window=thr))

                eval_list.append(
                    Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                                                                           casacade_version=True, hpf_strenght=hpf_strength,
                                                                           lee_filter_sigma_value=sigma,
                                                                           lee_sigma_filter_window=win_size, threshold_cliping_window=thr))

                # Selective UM
                t1 = 0.3; t2 = 0.8; bf_d = 19; ss = 40; T = 5
                eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, lambda_v=t1,
                                                                             lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss,
                                                                             bf_sigma_colors=ss, T=T))

                # Histogram Equalization UM
                str_1 = 0.225; str_2 = 0.025
                eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                                                                                          strength_1=str_1, strength_2=str_2))

        for el in eval_list:
            Application.do_histogram_job(port_input_name=el)
            Application.do_plot_lines_over_columns_job(port_input_name=el, lines=[50], columns=[240, 280])

        Application.create_config_file()
        Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
        # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
        # Application.run_application()

        for el in range(len(eval_list)):
            eval_list[el] += '_L0'

        # Benchmarking.run_PSNR_benchmark(input_location='Logs/application_results',
        #                                gt_location=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_original",
        #                                raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_original",
        #                                jobs_set=eval_list, db_calc=False)

        Utils.close_files()

        # suit_list = ['HPF', 'N_NUM', 'NUM', 'ANUM', 'AUM', 'C_CUM', 'CUM','SUM', 'HE_UM', 'UM']
        suit_list = ['NUM']

        for el in suit_list:
            sub_list_to_plot = ['GRAY_RAW_L0'] + [string for string in eval_list if string.startswith(el)]
            plot_multiple_lines_image(list_ports=sub_list_to_plot, plot_name=el, lines=[50], columns=[240, 270], name=el, img_name='test')


def plot_ratio_PSNR_runtime():
    import matplotlib.pyplot as plt
    # Data
    psnr_data = {
        "HPF": [33.160913, 32.405899, 33.352528, 29.524149, 32.847815, 28.223999, 31.399910, ],
        "UM": [33.059469, 33.141240, 33.080099, 32.971186, 33.170106, 29.361908, 33.337593, ],
        "NUM": [33.555355, 31.796908, 33.820938, 29.907309, 31.787842, 29.337985, 30.748512, ],
        "N_NUM": [32.958535, 32.992959, 33.026872, 33.072417, 33.132108, 33.152548, 33.236383, ],
        "ANUM": [34.033585, 32.985949, 34.226352, 24.376633, 33.602641, 17.548256, 31.752155, ],
        "AUM": [33.104559, 33.069507, 33.435460, 29.941666, 33.348578, 28.507332, 32.422046, ],
        "CUM": [33.515607, 34.548331, 34.062409, 39.222491, 34.742194, 40.100663, 35.817342, ],
        "C_CUM": [33.956022, 35.864402, 34.460376, 46.596228, 35.236987, 62.213203, 36.220378, ],
        "SUM": [33.058413, 33.316825, 33.180946, 30.699352, 33.483196, 28.344352, 33.632701, ],
        "HE_UM": [11.732178, 11.731171, 11.735056, 11.627060, 11.778433, 11.796022, 11.793492, ]
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
        for i,j in zip([r'.*(3x3_xy).*', r'.*(dilated_5x5_xy).*', r'.*(dilated_7x7_xy).*', r'.*(dilated_9x9_xy).*', r'.*(5x5_xy).*', r'.*(7x7_xy).*', r'.*(9x9_xy).*', r'.*(GRAY_RAW).*'],
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



def test_syntethic_img():
        """

        """
        Application.delete_folder_appl_out()
        Benchmarking.delete_folder_benchmark_out()

        Application.set_input_image_folder( r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt")


        raw = Application.do_get_image_job('RAW')
        grey = Application.do_grayscale_transform_job(port_input_name=raw)
        blurred = Application.do_gaussian_blur_image_job(port_input_name=grey, sigma=1.4)

        eval_list = list()

        eval_list.append(grey)
        eval_list.append(blurred)

        for (input, is_rgb) in ([blurred, False],):
            for kernel in [
                # CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2,
                CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,
                CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1,
                CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1
            ]:
                eval_list.append( Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, port_output_name='HPF_' + kernel))
                # Classical UM
                strength = 0.975
                eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength, port_output_name='UM_' + kernel))

                # Nonlinear UM
                gk = 0.4; hpf_strength = 0.025
                eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk,
                                                                             operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=kernel, hpf_strength=hpf_strength, port_output_name='NUM_' + kernel))

                # Normalized NUM
                strength = 0.975
                eval_list.append(Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength, port_output_name='N_NUM_' + kernel))

                # ANUM
                bf_d = 13;  bf_s = 30; bf_c = 50; t1 = 5; t2 = 45
                eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                                                                                       thr_1=t1, thr_2=t2, bf_distance=bf_d, bf_sigma_space=bf_s, bf_sigma_colors=bf_c, port_output_name='ANUM_' + kernel))
                # AUM
                t1 = 10; t2 = 40; dl = 2; dh = 3; mu = 0.075; beta = 0.1
                eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, thr_1=t1,
                                                                            thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta, port_output_name='AUM_' + kernel))

                # CUM
                hpf_strength = 0.45; win_size = 5; sigma = 10; thr = 3
                eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                                                                                        hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr,
                                                                                        port_output_name='CUM_' + kernel))

                # CCUM
                hpf_strength = 0.5; win_size = 3; sigma = 10; thr = 3
                eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                                                                                        casacade_version=True, hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma,
                                                                                        lee_sigma_filter_window=win_size, threshold_cliping_window=thr, port_output_name='C_CUM_' + kernel))

                # Selective UM
                t1 = 0.975; t2 = 0.025; bf_d = 11; sc = 20; ss=10; T = 7
                eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, lambda_v=t1,
                                                                         lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss, bf_sigma_colors=sc, T=T, port_output_name='SUM_' + kernel))

                # Histogram Equalization UM
                str_1 = 0.025; str_2 = 0.5
                eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                                                                                      strength_1=str_1, strength_2=str_2, port_output_name='HE_UM_' + kernel))

        # for el in eval_list:
        #     Application.do_histogram_job(port_input_name=el)
        for el in eval_list:
            Application.do_matrix_difference_job(port_input_name_1=grey, port_input_name_2=el, port_output_name= 'DIFF_' + el,
                                                 normalize_image=True, save_cmap=True, is_rgb=False)

        Application.create_config_file()
        Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
        # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
        Application.run_application()

        for el in range(len(eval_list)):
            eval_list[el] += '_L0'

        Benchmarking.run_PSNR_benchmark(input_location='Logs/application_results',
                                        gt_location=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt",
                                        raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt",
                                        jobs_set=eval_list, db_calc=False)
        #
        Benchmarking.run_SF_benchmark(input_location='Logs/application_results',
                                      raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt",
                                      jobs_set=eval_list)

        Benchmarking.run_Entropy_benchmark(input_location='Logs/application_results',
                                           raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt",
                                           jobs_set=eval_list)

        Benchmarking.run_RMSC_benchmark(input_location='Logs/application_results',
                                        raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt",
                                        jobs_set=eval_list)

        Benchmarking.run_BRISQUE_benchmark(input_location='Logs/application_results',
                                           raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt",
                                           jobs_set=eval_list)

        Utils.close_files()



def plot_ratio_PSNR_runtime_syn():
    import matplotlib.pyplot as plt
    # Data
    psnr_data = {
        "HPF": [29.629835, 30.649253, 32.198870, 21.477487, 30.479689, 17.229298, 28.315093, ],
        "UM": [29.562565, 30.852203, 32.154138, 21.550079, 30.597154, 17.261810, 28.445031, ],
        "NUM": [36.822336, 26.316218, 29.382894, 23.209104, 27.196875, 20.003729, 26.316198, ],
        "N_NUM": [28.536244, 28.447006, 28.314475, 28.272588, 28.139312, 28.149231, 28.029568, ],
        "ANUM": [31.493553, 32.439812, 34.535943, 12.557779, 32.611518, 10.213259, 29.758332, ],
        "AUM": [29.232492, 30.596185, 32.257460, 21.458798, 30.456366, 17.236630, 28.287529, ],
        "CUM": [28.414066, 33.978350, 30.201244, 42.700516, 31.261986, 39.826359, 31.715709, ],
        "C_CUM": [28.503548, 34.625323, 30.429829, 42.334777, 31.737495, 39.915251, 32.114293, ],
        "SUM": [29.562326, 30.852446, 32.153688, 21.550172, 30.597199, 17.268700, 28.495664, ],
        "HE_UM": [20.534535, 19.684356, 20.774946, 18.273801, 21.079900, 14.559124, 21.254403, ],
    }

    runtime_data = {
        "HPF": [6.25, 9.57, 6.12, 17.76, 5.99, 16.09, 6.99],
        "UM": [9.64, 10.77, 10.04, 13.24, 9.88, 15.16, 9.68],
        "NUM": [26.52, 26.96, 26.64, 28.88, 26.44, 32.36, 26.58],
        "N_NUM": [20.42, 21.32, 20.21, 23.03, 20.49, 25.80, 20.26],
        "ANUM": [1002.01, 1024.84, 1009.93, 1031.64, 1001.88, 1030.89, 1010.15],
        "AUM": [1170.15, 1175.10, 1174.71, 1173.44, 1164.15, 1180.56, 1171.79],
        "CUM": [257.40, 255.58, 258.46, 259.30, 255.45, 267.36, 254.62],
        "C_CUM": [258.55, 262.10, 261.05, 273.86, 258.44, 273.06, 266.49],
        "SUM": [1022.56, 1017.90, 1013.94, 1060.03, 995.88, 1015.79, 1025.85],
        "HE_UM": [220.66, 223.12, 224.48, 238.08, 224.03, 244.09, 225.61],
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

    plt.savefig(os.path.join(file_to_save, '{}.png'.format('syntethic_psnr_vs_runtime')), bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()

def test_natural_img():
    """

    """
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder(r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\orig")
    # Application.set_input_image_folder(r"c:\repos\eecvf_git\TestData\smoke_test")


    raw = Application.do_get_image_job('RAW')
    eval_list = list()
    eval_list.append(raw)

    for (input, is_rgb) in ([raw, True],):
        for kernel in [CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                       CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,
                       CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1,
                       CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1
                       ]:
            # for kernel in [CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1]:
            eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, port_output_name='HPF_' + kernel))
            # Classical UM
            strength = 0.525
            eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength, port_output_name='UM_' + kernel))
            # Nonlinear UM
            gk = 1.6; hpf_strength = 0.005
            eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk,
                                                                                     operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=kernel,
                                                                                     hpf_strength=hpf_strength, port_output_name='NUM_' + kernel))
            # Normalized NUM
            strength = 0.05
            eval_list.append(
                    Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength, port_output_name='N_NUM_' + kernel))
            #  ANUM
            bf_d = 9; bf_s = 40; bf_c = 20; t1 = 5; t2 = 20
            eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                                                                                                           thr_1=t1, thr_2=t2, bf_distance=bf_d,
                                                                                                           bf_sigma_space=bf_s, bf_sigma_colors=bf_c, port_output_name='ANUM_' + kernel))
            # AUM
            t1 = 5; t2 = 50; dl = 2; dh = 3; mu = 0.005; beta = 0.1
            eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, thr_1=t1,
                                                                                                        thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta, port_output_name='AUM_' + kernel))
            # CUM
            hpf_strength = 0.9; win_size = 11; sigma = 20; thr = 5
            eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                                                                                       hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma,
                                                                                       lee_sigma_filter_window=win_size, threshold_cliping_window=thr, port_output_name='CUM_' + kernel))
            # CCUM
            hpf_strength = 0.95; win_size = 11; sigma = 30; thr = 5
            eval_list.append(
                Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                                                                       casacade_version=True, hpf_strenght=hpf_strength,
                                                                       lee_filter_sigma_value=sigma,
                                                                       lee_sigma_filter_window=win_size, threshold_cliping_window=thr, port_output_name='C_CUM_' + kernel))
            # Selective UM
            t1 = 0.8; t2 = 0.025; bf_d = 9; sc=10; ss = 10; T = 3
            eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, lambda_v=t1,
                                                                         lambda_s=t2, bf_distance=bf_d, bf_sigma_space=ss,
                                                                         bf_sigma_colors=sc, T=T, port_output_name='SUM_' + kernel))

            # Histogram Equalization UM DDDD
            str_1 = 0.252; str_2 = 0.025
            eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                                                                                              strength_1=str_1, strength_2=str_2, port_output_name='HE_UM_' + kernel))
    for el in eval_list:
        # Application.do_histogram_job(port_input_name=el)
        Application.do_matrix_difference_job(port_input_name_1=raw, port_input_name_2=el, normalize_image=True, save_cmap=True, is_rgb=False, port_output_name= 'DIFF_' + el)
        Application.do_zoom_image_job(port_input_name=el, zoom_factor=1.5, do_interpolation=False, is_rgb=is_rgb, w_offset=75, port_output_name= 'ZOOM_' + el,)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
    Application.run_application()

    for el in range(len(eval_list)):
        eval_list[el] += '_L0'

    Benchmarking.run_SF_benchmark(input_location='Logs/application_results',
                                      raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\orig",
                                      jobs_set=eval_list)

    Benchmarking.run_Entropy_benchmark(input_location='Logs/application_results',
                                       raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\orig",
                                       jobs_set=eval_list)

    Benchmarking.run_RMSC_benchmark(input_location='Logs/application_results',
                                    raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\orig",
                                    jobs_set=eval_list)

    Benchmarking.run_BRISQUE_benchmark(input_location='Logs/application_results',
                                       raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\orig",
                                       jobs_set=eval_list)

    Utils.close_files()

def param_finder():
        """

        """
        Application.delete_folder_appl_out()
        Benchmarking.delete_folder_benchmark_out()

        # Application.set_input_image_folder( r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt")
        # Application.set_input_image_folder(r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_blur")
        Application.set_input_image_folder(r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\orig")

        raw = Application.do_get_image_job('RAW')
        grey = Application.do_grayscale_transform_job(port_input_name=raw)

        # blurred = Application.do_gaussian_blur_image_job(port_input_name=grey, sigma=1.4)

        eval_list = list()

        eval_list.append(grey)
        # eval_list.append(blurred)

        # kernels = [ CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1,  CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,
        #     CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1
        #     ]
        kernels = [
            CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1
        ]

        # for (input, is_rgb) in ([blurred, False],):
        # for (input, is_rgb) in ([raw, True],):
        #     for kernel in kernels:
                # Classical UM
                # for strength in range(25, 1000, 25):
                #     strength = strength / 1000
                #     eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength))
                # Nonlinear UM
                # for gk in range(4, 20, 2):
                #     gk = gk/10
                #     for hpf_strength in range(5,300,5):
                #         hpf_strength = hpf_strength / 1000
                #         eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk,
                #                                                                      operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=kernel,
                #                                                                      hpf_strength=hpf_strength))
                # Normalized NUM
                # for strength in range(5, 500, 5):
                #     strength = strength / 1000
                #     eval_list.append(
                #             Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength))
                # for bf_d in range(9, 13, 2):
                #     for bf_s in range(40, 60, 5):
                #         for bf_c in range(10, 30, 5):
                #             for t1 in range(5, 40, 5):
                #                 for t2 in range(20, 46, 5):
                #                     if t1<t2:
                #                         eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                #                                                                                                                    thr_1=t1, thr_2=t2, bf_distance=bf_d,
                #                                                                                                                    bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
                # AUM
                # for t1 in range(5, 20, 5):
                #     for t2 in range(40, 51, 5):
                #         if t1<t2:
                #             for dl in range(1, 3, 1):
                #                 for dh in range(3, 5, 1):
                #                     if dl<dh:
                #                         for mu in range(50, 800, 50):
                #                             mu = mu / 10000
                #                             for beta in range(10, 91, 10):
                #                                 beta = beta / 100
                #                                 eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb,
                #                                                                                             kernel=kernel, thr_1=t1,
                #                                                                                             thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
                # CUM
                # hpf_strength = 0.19; win_size = 9; sigma = 10; thr = 3
                # for hpf_strength in range(50, 1000, 50):
                #     hpf_strength = hpf_strength / 1000
                #     for win_size in range(3,13,2):
                #         for sigma in range(10, 40, 10):
                #             for thr in range(3, 11, 2):
                #                 eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                #                                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma,
                #                                                                                         lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
                # for hpf_strength in range(50, 1000, 50):
                #     hpf_strength = hpf_strength / 1000
                #     for win_size in range(3,13,2):
                #         for sigma in range(10, 40, 10):
                #             for thr in range(3, 11, 2):
                #                 eval_list.append(
                #                     Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                #                                                                            casacade_version=True, hpf_strenght=hpf_strength,
                #                                                                            lee_filter_sigma_value=sigma,
                #                                                                            lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
                # # Selective UM
                # t1 = 0.65; t2 = 0.05; bf_d = 17; ss = 30; T = 7
                # for t1 in range(800, 1000, 25):
                #     t1 = t1/1000
                #     for t2 in range(25, 200, 25):
                #         t2 = t2/1000
                #         for bf_d in range(9, 13, 2):
                #             for bf_s in range(10, 31, 10):
                #                 for bf_c in range(10, 31, 10):
                #                     for T in range(3, 4, 2):
                #                         eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, lambda_v=t1,
                #                                                                                      lambda_s=t2, bf_distance=bf_d, bf_sigma_space=bf_s,
                #                                                                                      bf_sigma_colors=bf_c, T=T))
                # Histogram Equalization UM DDDD
                # str_1 = 0.225; str_2 = 0.075
                # for str_1 in range(25, 1000, 25):
                #     str_1 = str_1/1000
                #     for str_2 in range(25, 1000, 25):
                #         str_2 = str_2/1000
                #         eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                #                                                                                   strength_1=str_1, strength_2=str_2))

        # # for synt image
                # # Classical UM
                # for strength in range(25, 1000, 25):
                #     strength = strength / 1000
                #     eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength))

                # # Nonlinear UM
                # for gk in range(4, 30, 2):
                #     for hpf_strength in range(25,1000,25):
                #         hpf_strength = hpf_strength / 1000
                #         eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk,
                #                                                                      operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=kernel,
                #                                                                      hpf_strength=hpf_strength))
                # # Normalized NUM
                # for strength in range(25, 1000, 25):
                #     strength = strength / 1000
                #     eval_list.append(
                #             Application.do_normalized_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, strenght=strength))
                #  ANUM
                # bf_d = 11; bf_s = 50; bf_c = 20; t1 = 25; t2 = 30
                # for bf_d in range(9, 14, 2):
                #     for bf_s in range(30, 51, 10):
                #         for bf_c in range(20, 51, 10):
                #             for t1 in range(5, 40, 5):
                #                 for t2 in range(20, 46, 5):
                #                     if t1<t2:
                #                         eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                #                                                                                                                    thr_1=t1, thr_2=t2, bf_distance=bf_d,
                #                                                                                                                    bf_sigma_space=bf_s, bf_sigma_colors=bf_c))
                # AUM
                # t1 = 10; t2 = 45; dl = 2; dh = 3; mu = 0.01; beta = 0.1
                # #t1 = 10
                # for t1 in range(10, 21, 5):
                #     for t2 in range(40, 56, 5):
                #         if t1<t2:
                #             for dl in range(1, 3, 1):
                #                 for dh in range(3, 5, 1):
                #                     if dl<dh:
                #                         for mu in range(5, 300, 10):
                #                             mu = mu / 1000
                #                             for beta in range(10, 91, 10):
                #                                 beta = beta / 100
                #                                 eval_list.append(Application.do_adaptive_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, thr_1=t1,
                #                                                                                                                         thr_2=t2, alpha_low=dl, alpha_high=dh, mu=mu, beta=beta))
                # CUM
                # hpf_strength = 0.19; win_size = 9; sigma = 10; thr = 3
                # for hpf_strength in range(50, 1000, 50):
                #     hpf_strength = hpf_strength / 1000
                #     for win_size in range(3,11,2):
                #         for sigma in range(10, 40, 10):
                #             for thr in range(3, 11, 2):
                #                 eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                #                                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma,
                #                                                                                         lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
                # CCUM
                # for hpf_strength in range(50, 1000, 50):
                #     hpf_strength = hpf_strength / 1000
                #     for win_size in range(3,11,2):
                #         for sigma in range(10, 40, 10):
                #             for thr in range(3, 11, 2):
                #                 eval_list.append(
                #                     Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=kernel,
                #                                                                            casacade_version=True, hpf_strenght=hpf_strength,
                #                                                                            lee_filter_sigma_value=sigma,
                #                                                                            lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
                # # Selective UM
                # t1 = 0.65; t2 = 0.05; bf_d = 17; ss = 30; T = 7
                # for t1 in range(800, 1000, 25):
                #     t1 = t1/1000
                #     for t2 in range(25, 200, 25):
                #         t2 = t2/1000
                #         for bf_d in range(9, 13, 2):
                #             for bf_s in range(10, 31, 10):
                #                 for bf_c in range(10, 31, 10):
                #                     for T in range(3, 4, 2):
                #                         eval_list.append(Application.do_selective_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel, lambda_v=t1,
                #                                                                                      lambda_s=t2, bf_distance=bf_d, bf_sigma_space=bf_s,
                #                                                                                      bf_sigma_colors=bf_c, T=T))
                # # Histogram Equalization UM DDDD
                # str_1 = 0.225; str_2 = 0.075
                # for str_1 in range(25, 1000, 25):
                #     str_1 = str_1/1000
                #     for str_2 in range(25, 1000, 25):
                #         str_2 = str_2/1000
                #         eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=kernel,
                #                                                                                   strength_1=str_1, strength_2=str_2))

        Application.create_config_file()
        Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
        # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
        Application.run_application()

        for el in range(len(eval_list)):
            eval_list[el] += '_L0'

        # Benchmarking.run_PSNR_benchmark(input_location='Logs/application_results',
        #                                 gt_location=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt",
        #                                 raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_synt",
        #                                 jobs_set=eval_list, db_calc=False)
        #
        # Benchmarking.run_PSNR_benchmark(input_location='Logs/application_results',
        #                                gt_location=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_original",
        #                                raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\imgs\test_img_original",
        #                                jobs_set=eval_list, db_calc=False)

        Benchmarking.run_PSNR_benchmark(input_location='Logs/application_results',
                                       gt_location=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\orig",
                                       raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\enh",
                                       jobs_set=eval_list, db_calc=False)

        Benchmarking.run_BRISQUE_benchmark(input_location='Logs/application_results',
                                           raw_image=r"c:\Documente\faculta\publicatii\2023_The effect of dilated filters in sharpening algorithms\data\natural_img\orig",
                                           jobs_set=eval_list)

        for idx in range(len(kernels)):
            kernels[idx] = kernels[idx].lower()

        # suit_list = ['HPF', 'N_NUM', 'NUM', 'ANUM', 'AUM', 'C_CUM', 'CUM','SUM', 'HE_UM', 'UM']
        suit_list = ['HE_UM']

        for el in suit_list:
            Utils.plot_box_benchmark_values(name_to_save='PSNR_' + el, number_decimal=3, title_name=el,
                                            show_plot=False, save_plot=True,
                                            data_subsets=kernels,
                                            data='PSNR', eval=[string for string in eval_list if string.startswith(el)])

            Utils.plot_box_benchmark_values(name_to_save='BRISQUE_' + el, number_decimal=3, title_name=el,
                                            show_plot=False, save_plot=True,
                                            data_subsets=kernels,
                                            data='BRISQUE', eval=[string for string in eval_list if string.startswith(el)])
        Utils.close_files()




if __name__ == "__main__":
    # test_single_line_effect()
    # plot_ratio_PSNR_runtime()
    # test_syntethic_img()
    # plot_ratio_PSNR_runtime_syn()
    test_natural_img()
    # param_finder()