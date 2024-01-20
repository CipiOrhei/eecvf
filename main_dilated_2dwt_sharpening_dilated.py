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
  title={An Image Sharpening Technique Based on Dilated Filters and 2D-DWT Image Fusion},
  author={Bogdan, Victor and Bonchis, Bogdan and Orhei, Ciprian},
  booktitle={In Proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2024)},
  pages={--},
  year={2024},
  organization={--}

"""
# Custom plot
def plot_frame_values(name_to_save: str, eval: list, data,
                      number_decimal: int = 3, set_name_replace_list=None,
                      x_label_font_size=40, y_label_font_size=40, x_ticks_font_size=30, y_ticks_font_size=30, dpi_save_value=300,
                      title_font_size=40, img_size_w=15, img_size_h=10, legend_font_size='medium', legend_name='Jobs', title_name=None,
                      save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = True):
    """
      Plot custom ports
        :param name_to_save: name you want for plot
        :param eval: list of ports evaluated to plot
        :param number_decimal: number of decimals to use
        :param set_name_replace_list: list of string to replace for labels in legend
        :param y_plot_name: name of y label
        :param x_label_font_size font size of x axis label
        :param y_label_font_size font size of y axis label
        :param x_ticks_font_size font size of x axis ticks
        :param y_ticks_font_size font size of y axis ticks
        :param title_font_size font size of plot title
        :param title_name name of plot title
        :param legend_font_size size of legend font (xx-small, x-small, small, medium, large, x-large, xx-large)
        :param legend_name name of legend
        :param img_size_w save width image
        :param img_size_h save height image
        :param dpi_save_value save dpi of image
                The bigger value the longer time it takes to save
       :param show_plot: if we want to show the plot
       :param save_plot: if we want to save the plot
       :param save_location: where to save
       :return None
      """

    import matplotlib.pyplot as plt
    import os

    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, data)
    subset_dict = dict()

    # get files from benchmark folder
    for dirname, dirnames, filenames in os.walk(input_location):
        for filename in filenames:
            # files.append(filename)
            try:
            # if True:
                f = open(os.path.join(input_location, filename)).readlines()
                set_name = filename.split('.')[0]

                if set_name in eval:

                    if set_name_replace_list is not None:
                        for el in set_name_replace_list:
                            set_name = set_name.replace(el[0], el[1])

                    subset_dict[set_name] = {'average': 0, 'frames': list(), 'values': list()}

                    for line in range(len(f)):
                        if line == 0:
                            pass
                        elif line == (len(f)-1):
                            tmp = f[line].split(' ')
                            for el in range(len(tmp)-1, 0, -1):
                                if tmp[el] != '' and tmp[el] != ' ' and tmp[el] != '\n':
                                    subset_dict[set_name]['average'] = round(float(tmp[el]), number_decimal)
                                    break

                        else:
                            tmp = f[line].split(' ')
                            subset_dict[set_name]['frames'].append(tmp[0])
                            for el in range(len(tmp)-1, 0, -1):
                                if tmp[el] != '' and tmp[el] != ' ' and tmp[el] != '\n':
                                    subset_dict[set_name]['values'].append(round(float(tmp[el]), number_decimal))
                                    break

            except:
                print(filename, 'NOK TO USE FOR PLOTTING')

    tst = dict()
    for frame in subset_dict['RAW']['frames']:
        tst[frame] = 0

    for set in subset_dict.keys():
        for frame_idx in range(len(subset_dict[set]['frames'])):
            tst[subset_dict[set]['frames'][frame_idx]] += subset_dict[set]['values'][frame_idx]

    for idx in tst.keys():
        tst[idx] /=10

    subset_dict['AVERAGE'] = {'average': 0, 'frames': list(tst.keys()), 'values': list(tst.values())}

    for set in subset_dict.keys():
        if set in ['AVERAGE']:
            plt.plot(subset_dict[set]['frames'], subset_dict[set]['values'], marker='_', linestyle='', ms=20,  label=set)
        elif set in ['UM_P_AVG_1', 'UM_P_AVG_2', 'UM_P_AVG_3', 'UM_P_MAX']:
            plt.plot(subset_dict[set]['frames'], subset_dict[set]['values'], marker='D', linestyle='',ms=10,label=set)
        else:
            plt.plot(subset_dict[set]['frames'], subset_dict[set]['values'], marker='.', linestyle='',ms=10,label=set)

    # plt.plot('AVERAGE', tst[idx], marker='_', linestyle='', label=set)

    fig = plt.gcf()
    fig.set_size_inches(w=img_size_w, h=img_size_h)
    plt.xlabel('Frames', fontsize=x_label_font_size)
    plt.ylabel(data, fontsize=y_label_font_size)
    plt.yticks(fontsize=y_ticks_font_size)
    plt.xticks(fontsize=x_ticks_font_size)
    # plt.legend(fancybox=True, fontsize=legend_font_size, loc='best', title=legend_name)
    plt.legend(fancybox=True,fontsize=legend_font_size,title=legend_name, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, mode="expand", ncol=6)
    # plt.legend(fancybox=True, fontsize=legend_font_size, loc='best', title=legend_name)

    if title_name is not None:
        plt.title(title_name, fontsize=title_font_size)

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, name_to_save + '.png'), bbox_inches='tight', dpi=dpi_save_value)

    plt.clf()
    plt.close()



def main_paper():
    """

    """
    Application.delete_folder_appl_out()
    # Please change accordingly to data you wish to execute on
    Application.set_input_image_folder('TestData/sharpnnes_test')
    # Application.set_input_image_folder('TestData/smoke_test')
    # Application.set_input_image_folder('TestData/sharpnnes_test2')
    # Application.set_input_image_folder('TestData/TMBuD/images')

    raw = Application.do_get_image_job('RAW')

    eval_list = list()


    eval_list.append(raw)

    input = raw
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
            plot_frame_values(name_to_save=data + '_' + el[0], eval=el[1], data=data, set_name_replace_list=el[2], save_plot=True,
                              x_label_font_size=30, y_label_font_size=30, x_ticks_font_size=20, y_ticks_font_size=20,
                              legend_name=None, legend_font_size='medium', dpi_save_value=800)
            # Only when TMBuD or other big dataset
            # Utils.plot_box_benchmark_values(name_to_save=data + '_box', number_decimal=3,
            #                                 data=data, data_subsets=el[1], eval=eval_list)

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