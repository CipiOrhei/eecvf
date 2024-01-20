import os.path
import numpy as np
import cv2
import math

import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils


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
        for i,j in zip([r'.*(HPF).*', r'UM.*(dilated_5x5_xy).*', r'UM.*(dilated_7x7_xy).*', r'UM.*(3x3).*', r'.*NUM.*', r'.*(ANUM).*', r'.*(CUM_D_2DWT).*', r'.*(CUM).*', r'.*(HE).*',r'.*(GRAY_RAW).*', '_L0'],
                       ['HPF','UM_5D','UM_7D','UM','NUM','ANUM','D_CUM_2DWT', 'CUM', 'HE', 'original', '']):
            name = re.sub(i, j, port)
            # data_series_names.append(name)
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
                                # title_name=plot_name, title_font_size=18,
                                data_series_names=data_series_names, legend_name=None, show_legend=True, if_grid=True,
                                name_folder='PLOT_LINE_suits',
                                name_to_save='plot_line_{}'.format(plot_name),
                                save_plot=True, show_plot=False)

def test_single_line_effect():
    """

    """
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder(r"E:\repo\eecvf\TestData\binary_img\blur")
    # Application.set_input_image_folder('TestData/smoke_test')
    raw = Application.do_get_image_job('RAW')
    grey = Application.do_grayscale_transform_job(port_input_name=raw)

    eval_list = list()
    eval_list.append(grey)

    for (input, is_rgb) in ([grey, False],):
        # Classical sharpening with HPF
        eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1))
        # Classical UM
        eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,strenght=0.1))
        eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,strenght=0.1))
        eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1,strenght=0.1))
        # Nonlinear UM
        eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=0.4,
                                                                     operator=CONFIG.FILTERS.SOBEL_3x3, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, hpf_strength=0.025))
        # ANUM
        eval_list.append(Application.do_adaptive_non_linear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                                                                                   thr_1=5, thr_2=60, bf_distance=19, bf_sigma_space=30, bf_sigma_colors=30))
        # CUM
        eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                                                                                hpf_strenght=0.7, lee_filter_sigma_value=35, lee_sigma_filter_window=7, threshold_cliping_window=5))
        # Histogram Equalization UM
        # eval_list.append(Application.do_histogram_equalization_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
        #                                                                           strength_1=0.225, strength_2=0.025))


        eval_list.append(Application.do_um_2dwt_fusion(port_input_name=input, is_rgb=is_rgb, octaves=3, s=1.3, k=math.sqrt(2), m=1, wavelet='haar', port_output_name='UM_2DWT'))


        # eval_list.append(Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
        #                                                        strenght=1.9, levels_fusion=3, fusion_rule='max', port_output_name='UM_D_2DWT_MAX'))
        eval_list.append(Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
                                                               strenght=1.9, levels_fusion=3, fusion_rule='average_1', port_output_name='UM_D_2DWT_AVG1'))
        # eval_list.append(Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
        #                                                        strenght=1.9, levels_fusion=3, fusion_rule='average_2', port_output_name='UM_D_2DWT_AVG2'))
        # eval_list.append(Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
        #                                                        strenght=1.9, levels_fusion=3, fusion_rule='average_3', port_output_name='UM_D_2DWT_AVG3'))

        eval_list.append(
            Application.do_constrained_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb,
                                                                       kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                                                                       wave_lenght='db4',
                                                                       strenght=1.9, levels_fusion=2, fusion_rule='average_1',
                                                                       # port_output_name='CUM_D_2DWT_AVG3',
                                                                       lee_filter_sigma_value=5,
                                                                       lee_sigma_filter_window=7,
                                                                       threshold_cliping_window=3))

        # for s in range(5, 2000, 5):
        #     for n in [2,3,4,5]:
        #         for lw in [7]:
        #             for ls in [5]:
        #                 for th in [3, 5, 7]:
        #                     for a in ['max', 'average_1','average_2', 'average_3']:
                            # for a in ['average_1']:
                            #     eval_list.append(Application.do_constrained_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
                            #                                                                                 strenght=s/100, levels_fusion=n, fusion_rule=a,
                            #                                                                                 port_output_name='CUM_D_2DWT_AVG3',
                            #                                                                                 lee_filter_sigma_value=ls, lee_sigma_filter_window=lw, threshold_cliping_window=th))
        # for s in range(50, 10000, 50):
        #     eval_list.append(
        #                         Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb,
        #                                                                        kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
        #                                                                        wave_lenght='db4',
        #                                                                        strenght=s/1000, levels_fusion=3,
        #                                                                        fusion_rule='average_1'))


    for el in eval_list:
        # Application.do_histogram_job(port_input_name=el)
        Application.do_plot_lines_over_columns_job(port_input_name=el, lines=[50], columns=[240, 280])

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
    Application.run_application()

    for el in range(len(eval_list)):
        eval_list[el] += '_L0'

    Benchmarking.run_PSNR_benchmark(input_location='Logs/application_results',
                                   gt_location=r"E:\repo\eecvf\TestData\binary_img\blur",
                                   raw_image=r"E:\repo\eecvf\TestData\binary_img\org",
                                   jobs_set=eval_list, db_calc=False)

    Utils.close_files()

    # suit_list = ['HPF', 'N_NUM', 'NUM', 'ANUM', 'AUM', 'C_CUM', 'CUM','SUM', 'HE_UM', 'UM']
    # suit_list = ['NUM']

    # for el in suit_list:
    #     # sub_list_to_plot = ['GRAY_RAW_L0'] + [string for string in eval_list if string.startswith(el)]
    #     sub_list_to_plot = eval_list
    for el in eval_list:
        if el != 'GRAY_RAW_L0':
            sub_list_to_plot = ['GRAY_RAW_L0'] + [el]
            plot_multiple_lines_image(list_ports=sub_list_to_plot, plot_name='results_'+el, lines=[50], columns=[240, 270], img_name='test', name='test')

    # plot_multiple_lines_image(list_ports=eval_list, plot_name='results', lines=[50], columns=[240, 270], name='results',
    #                               img_name='test')


if __name__ == "__main__":
    test_single_line_effect()