import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils


def main_paper():
    """

    """
    Application.delete_folder_appl_out()
    Application.set_input_image_folder('TestData/sharpnnes_test2')
    raw = Application.do_get_image_job('RAW')
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')

    eval_list = list()

    # eval_list.append(raw)
    eval_list.append(grey)

    for (input, is_rgb) in ([grey, False],):
        # for w in ['haar', 'db4', 'bior3.5']:
        #     for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        #         for lv in [2, 3, 4, 5]:
        eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=0.7))
        eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strenght=0.7))
        eval_list.append(Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                                                                            wave_lenght='haar', strenght=0.9, levels_fusion=3))
            #         eval_list.append(Application.do_unsharp_filter_dilated_2dwt_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
            #                                                                         wave_lenght=w, strenght=s, levels_fusion=lv))

    for el in eval_list:
        Application.do_histogram_job(port_input_name=el)
        Application.do_mean_pixel_image_job(port_input_name=el)
        # Application.do_zoom_image_job(port_input_name=el, zoom_factor=2, do_interpolation=True, is_rgb=False)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
    Application.run_application()

    for el in range(len(eval_list)):
        eval_list[el] += '_L0'

    Benchmarking.run_SF_benchmark(input_location='Logs/application_results',
                                   raw_image='TestData/sharpnnes_test2',
                                   jobs_set=eval_list)

    Benchmarking.run_Entropy_benchmark(input_location='Logs/application_results',
                                       raw_image='TestData/sharpnnes_test2',
                                       jobs_set=eval_list)

    Benchmarking.run_RMSC_benchmark(input_location='Logs/application_results',
                                    raw_image='TestData/sharpnnes_test2',
                                    jobs_set=eval_list)

    Benchmarking.run_BRISQUE_benchmark(input_location='Logs/application_results',
                                    raw_image='TestData/sharpnnes_test2',
                                    jobs_set=eval_list)

    # list(name, list_to_eval, list_to_replace)
    list_to_plot = [
        ('UM_V1_GRAY', [set for set in eval_list if ('UNSHARP_FILTER' in set and 'v1' in set and 'GRAY_RAW' in set) or ('UM' in set and 'v1' in set and 'GRAY_RAW' in set) or set == 'GRAY_RAW_L0'], [('UNSHARP_FILTER', 'UM_'), ('laplace_v1', 'V1'), ('_xy_S_0_7_GRAY_RAW_L0', '')]),
    ]
    #
    for data in ['Entropy', 'SF', 'RMSC', 'BRISQUE']:
        for el in list_to_plot:
            Utils.plot_frame_values(name_to_save=data + '_' + el[0], eval=el[1], data=data, set_name_replace_list=el[2], save_plot=True,
                                    x_label_font_size=30, y_label_font_size=30, x_ticks_font_size=20, y_ticks_font_size=20,
                                    legend_name=None, legend_font_size='medium', dpi_save_value=800)

    # for el in list_to_plot:
    #     new_port_list = list()
    #     for el_port in el[1]:
    #         new_port_list.append('MEAN PX ' + el_port)
    #
    #     Utils.plot_custom_list(port_list=new_port_list, set_frame_name=True, set_name_replace_list=el[2],
    #                            name_to_save='MEAN_Px_' + el[0], y_plot_name='Pixel Value',
    #                            x_label_font_size=30, y_label_font_size=30, x_ticks_font_size=20, y_ticks_font_size=20,
    #                            legend_name=None, legend_font_size='medium', dpi_save_value=800,
    #                            show_plot=False, save_plot=True)

    Utils.close_files()


if __name__ == "__main__":
    main_paper()