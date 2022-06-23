import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils

import math

"""
This module contains the code used for the following paper:
  title={An image sharpening technique based on dilated filters and 2D-DWT image fusion},
  author={Bogdan, Victor and Orhei, Ciprian and Bonchis, Bogdan},
  booktitle={he Second International Conference on Computational Methods and Applications in Engineerin (ICCMAE) 2022},
  pages={--},
  year={2022},
  organization={--}

"""

def main_paper():
    """

    """
    Application.delete_folder_appl_out()

    Application.set_input_image_folder('TestData/sharpnnes_test')
    # Application.set_input_image_folder('TestData/smoke_test')
    # Application.set_input_image_folder('TestData/sharpnnes_test2')
    # Application.set_input_image_folder('TestData/TMBuD/images')
    raw = Application.do_get_image_job('RAW')
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')

    eval_list = list()

    # grey_blur = Application.do_add_gaussian_blur_noise_job(port_input_name=raw, mean_value=0.5, is_rgb=True)
    # grey_blur = Application.do_add_gaussian_blur_noise_job(port_input_name=grey, mean_value=0.5, is_rgb=False)
    # eval_list.append(grey_blur)

    eval_list.append(raw)
    # eval_list.append(grey)
    # input = grey
    input = raw
    # is_rgb = False
    is_rgb = True

    um_standard = Application.do_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, port_output_name='UM_STD')
    eval_list.append(um_standard)
    Application.do_matrix_difference_job(port_input_name_1=um_standard, port_input_name_2=input, normalize_image=True, save_cmap=True)

    um_std_laplace = Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=0.7, port_output_name='UM_STD_LAPLACE')
    eval_list.append(um_std_laplace)
    Application.do_matrix_difference_job(port_input_name_1=um_std_laplace, port_input_name_2=input, normalize_image=True, save_cmap=True, is_rgb=is_rgb)

    um_5d = Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strenght=0.7, port_output_name='UM_5D')
    eval_list.append(um_5d)
    Application.do_matrix_difference_job(port_input_name_1=um_5d, port_input_name_2=input, normalize_image=True, save_cmap=True, is_rgb=is_rgb)

    um_7d = Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, strenght=0.7, port_output_name='UM_7D')
    eval_list.append(um_7d)
    Application.do_matrix_difference_job(port_input_name_1=um_7d, port_input_name_2=input, normalize_image=True, save_cmap=True, is_rgb=is_rgb)

    um_2dwt = Application.do_um_2dwt_fusion(port_input_name=input, is_rgb=is_rgb, octaves=3, s=1.3, k=math.sqrt(2), m=1, wavelet='haar', port_output_name='UM_2DWT')
    eval_list.append(um_2dwt)
    Application.do_matrix_difference_job(port_input_name_1=um_2dwt, port_input_name_2=input, normalize_image=True, is_rgb=is_rgb, save_cmap=True)

    um_d_2dwt_0 = Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
                                                               strenght=0.7, levels_fusion=3, fusion_rule='max', port_output_name='UM_PROPOSED_MAX')
    eval_list.append(um_d_2dwt_0)
    Application.do_matrix_difference_job(port_input_name_1=um_d_2dwt_0, port_input_name_2=input, normalize_image=True, save_cmap=True, is_rgb=is_rgb)

    um_d_2dwt_1 = Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
                                                                 strenght=0.7, levels_fusion=3, fusion_rule='average_1', port_output_name='UM_PROPOSED_AVG_1')
    eval_list.append(um_d_2dwt_1)
    Application.do_matrix_difference_job(port_input_name_1=um_d_2dwt_1, port_input_name_2=input, normalize_image=True, save_cmap=True, is_rgb=is_rgb)

    um_d_2dwt_2 = Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
                                                               strenght=0.7, levels_fusion=3, fusion_rule='average_2', port_output_name='UM_PROPOSED_AVG_2')
    eval_list.append(um_d_2dwt_2)
    Application.do_matrix_difference_job(port_input_name_1=um_d_2dwt_2, port_input_name_2=input, normalize_image=True, save_cmap=True, is_rgb=is_rgb)

    um_d_2dwt_3 = Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, wave_lenght='db4',
                                                               strenght=0.7, levels_fusion=3, fusion_rule='average_3', port_output_name='UM_PROPOSED_AVG_3')
    eval_list.append(um_d_2dwt_3)
    Application.do_matrix_difference_job(port_input_name_1=um_d_2dwt_3, port_input_name_2=input, normalize_image=True, save_cmap=True, is_rgb=is_rgb)

    for el in eval_list:
        Application.do_histogram_job(port_input_name=el)
        Application.do_mean_pixel_image_job(port_input_name=el)
        Application.do_zoom_image_job(port_input_name=el, zoom_factor=1.5, do_interpolation=False, is_rgb=is_rgb, w_offset=75)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
    Application.run_application()

    for el in range(len(eval_list)):
        eval_list[el] += '_L0'

    Benchmarking.run_SF_benchmark(input_location='Logs/application_results',
                                   raw_image='TestData/sharpnnes_test',
                                   # raw_image='TestData/TMBuD/images',
                                   jobs_set=eval_list)

    Benchmarking.run_Entropy_benchmark(input_location='Logs/application_results',
                                       raw_image='TestData/sharpnnes_test',
                                       # raw_image='TestData/TMBuD/images',
                                       jobs_set=eval_list)

    Benchmarking.run_RMSC_benchmark(input_location='Logs/application_results',
                                    raw_image='TestData/sharpnnes_test',
                                    # raw_image='TestData/TMBuD/images',
                                    jobs_set=eval_list)

    Benchmarking.run_BRISQUE_benchmark(input_location='Logs/application_results',
                                       raw_image='TestData/sharpnnes_test',
                                       # raw_image='TestData/sharpnnes_test2',
                                       # raw_image='TestData/TMBuD/images',
                                    jobs_set=eval_list)

    # list(name, list_to_eval, list_to_replace)
    list_to_plot = [
        ('UM_V1_GRAY',
         [set for set in eval_list if ('UM_' in set) or set == 'GRAY_RAW_L0' or set == 'RAW_L0'],
         [('UNSHARP_FILTER', 'UM_'), ('laplace_v1', 'V1'), ('_xy_S_0_7_GRAY_RAW_L0', '')]),
    ]
    #
    for data in ['Entropy', 'SF', 'RMSC', 'BRISQUE']:
        for el in list_to_plot:
            Utils.plot_frame_values(name_to_save=data + '_' + el[0], eval=el[1], data=data, set_name_replace_list=el[2], save_plot=True,
                                    x_label_font_size=30, y_label_font_size=30, x_ticks_font_size=20, y_ticks_font_size=20,
                                    legend_name=None, legend_font_size='medium', dpi_save_value=800)

    Utils.close_files()

    for el in list_to_plot:
        new_port_list = list()
        for el_port in el[1]:
            new_port_list.append('MEAN PX ' + el_port)

        Utils.plot_custom_list(port_list=new_port_list, set_frame_name=True, set_name_replace_list=el[2],
                               name_to_save='MEAN_Px_' + el[0], y_plot_name='Pixel Value',
                               x_label_font_size=30, y_label_font_size=30, x_ticks_font_size=20, y_ticks_font_size=20,
                               legend_name=None, legend_font_size='medium', dpi_save_value=800,
                               show_plot=False, save_plot=True)

    Utils.close_files()


if __name__ == "__main__":
    main_paper()