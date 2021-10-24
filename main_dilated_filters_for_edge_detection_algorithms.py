# noinspection PyUnresolvedReferences
import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils


# def main_find_thr_first_order_edges(dataset):
#     # Application.delete_folder_appl_out()
#     # Benchmarking.delete_folder_benchmark_out()
#
#     Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)
#
#     list_to_save = []
#
#     Application.do_get_image_job(port_output_name='RAW')
#     Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')
#
#     # find best threshold for first level
#     for thr in range(15, 255, 10):
#         print('thr=', thr)
#         edge_result = Application.do_first_order_derivative_operators(port_input_name='GREY', operator=CONFIG.FILTERS.SOBEL_3x3)
#
#         thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
#                                                              input_threshold_type='cv2.THRESH_BINARY',
#                                                              port_output_name='THR_' + str(thr) + '_' + edge_result)
#         thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
#                                                                           port_output_name='FINAL_' + thr_edge_result)
#         list_to_save.append(thin_thr_edge_result + '_L0')
#
#     Application.create_config_file()
#     Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
#     # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
#     # Application.run_application()
#
#     # Do bsds benchmarking
#     # Be ware not to activate job_name_in_port in Application.configure_save_pictures
#     # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
#     #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
#     #                                             raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
#     #                                             jobs_set=list_to_save, do_thinning=False)
#
#     Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='first_order_thr_results_finder',
#                                  prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_SOBEL_3x3_GREY_L0',
#                                  list_of_data=list_to_save, number_of_series=50,
#                                  replace_list=[('THR_', 'Thr=')],
#                                  inputs=[''], self_contained_list=True,
#                                  save_plot=True, show_plot=False)
#
#     Utils.close_files()


# def main_find_sigma_first_order_edges(dataset):
#     Application.delete_folder_appl_out()
#     Benchmarking.delete_folder_benchmark_out()
#
#     Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)
#
#     list_to_save = []
#     list_to_benchmark = []
#     threshold = 65
#
#     Application.do_get_image_job(port_output_name='RAW')
#     Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')
#
#     edge = CONFIG.FILTERS.SOBEL_3x3
#
#     # find best threshold for first level
#     for sigma in range(25, 500, 25):
#         s = sigma / 100
#         blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
#                                                             port_output_name='BLURED_S_' + str(s).replace('.', '_'))
#         edge_result = Application.do_first_order_derivative_operators(port_input_name=blured_img, operator=edge)
#         thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=threshold,
#                                                              input_threshold_type='cv2.THRESH_BINARY',
#                                                              port_output_name='THR_' + edge_result)
#         thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
#                                                                           port_output_name='FINAL_' + edge_result)
#         list_to_save.append(thin_thr_edge_result + '_L0')
#
#     Application.create_config_file()
#     Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
#     # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
#     Application.run_application()
#
#     # Do bsds benchmarking
#     # Be ware not to activate job_name_in_port in Application.configure_save_pictures
#     Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
#                                                 gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
#                                                 raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
#                                                 jobs_set=list_to_save, do_thinning=False)
#
#     Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='first_order_sigma_results_finder',
#                                  prefix_to_cut_legend='FINAL_SOBEL_3x3_', suffix_to_cut_legend='_L0',
#                                  list_of_data=list_to_save, number_of_series=50,
#                                  replace_list=[('BLURED_S_', 'S='), ('_', '.')],
#                                  inputs=[''], self_contained_list=True,
#                                  save_plot=True, show_plot=False)
#
#     Utils.close_files()


def main_find_param_first_order_edges(dataset):
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    edges = [
        CONFIG.FILTERS.SOBEL_3x3,
        CONFIG.FILTERS.SOBEL_5x5,
        CONFIG.FILTERS.SOBEL_7x7,
        CONFIG.FILTERS.SOBEL_DILATED_5x5,
        CONFIG.FILTERS.SOBEL_DILATED_7x7
             ]

    for edge in edges:
        # find best threshold for first level
        for thr in range(30, 160, 10):
            # for thr in [10]:
            for sigma in range(25, 300, 25):
                # for sigma in [200]:
                s = sigma / 100
                # print('thr=', thr)
                blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                                    port_output_name='BLURED_S_' + str(s).replace('.', '_'))
                edge_result = Application.do_first_order_derivative_operators(port_input_name=blured_img,
                                                                              operator=edge)

                thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                     input_threshold_type='cv2.THRESH_BINARY',
                                                                     port_output_name='THR_' + str(thr) + '_' + edge_result)
                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                                  port_output_name='FINAL_' + thr_edge_result)
                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
    #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
    #                                             raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
    #                                             jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='first_order_thr_sigma_param_finder',
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_L0',
                                 list_of_data=list_to_save, number_of_series=40,
                                 replace_list=[('_SOBEL', ''),('_DILATED', ' Dilated '),
                                               ('_3x3', ' 3x3'), ('_5x5', ' 5x5'), ('_7x7', ' 7x7'), ('THR_', ' Thr='), ('_BLURED_S_', ' S='), ('_', '.')],
                                 inputs=[''], self_contained_list=True,set_legend_left=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_first_order_edge_detection(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    first_order_edge = [
        CONFIG.FILTERS.PIXEL_DIFF_3x3, CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_3x3
        , CONFIG.FILTERS.SOBEL_3x3
        , CONFIG.FILTERS.PREWITT_3x3
        , CONFIG.FILTERS.KIRSCH_3x3
        , CONFIG.FILTERS.KITCHEN_MALIN_3x3
        , CONFIG.FILTERS.KAYYALI_3x3
        , CONFIG.FILTERS.SCHARR_3x3
        , CONFIG.FILTERS.KROON_3x3
        , CONFIG.FILTERS.ORHEI_3x3
    ]

    threshold = 50
    sigma = 2.75

    for edge in first_order_edge:
        ########################################################################################################################
        # First order edge detection magnitude
        ########################################################################################################################
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=sigma)
        edge_result = Application.do_first_order_derivative_operators(port_input_name=blured_img, operator=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=threshold,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')

    first_order_edge = [
        CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_5x5
        , CONFIG.FILTERS.PIXEL_DIFF_5x5

        , CONFIG.FILTERS.SOBEL_5x5
        , CONFIG.FILTERS.PREWITT_5x5
        , CONFIG.FILTERS.KIRSCH_5x5
        , CONFIG.FILTERS.SCHARR_5x5
        , CONFIG.FILTERS.ORHEI_B_5x5
    ]

    threshold = 50
    sigma = 2.5

    for edge in first_order_edge:
        ########################################################################################################################
        # First order edge detection magnitude
        ########################################################################################################################
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=sigma)
        edge_result = Application.do_first_order_derivative_operators(port_input_name=blured_img, operator=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=threshold,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_7x7
        , CONFIG.FILTERS.PREWITT_7x7
    ]

    threshold = 30
    sigma = 2.75

    for edge in first_order_edge:
        ########################################################################################################################
        # First order edge detection magnitude
        ########################################################################################################################
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=sigma)
        edge_result = Application.do_first_order_derivative_operators(port_input_name=blured_img, operator=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=threshold,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')

    first_order_edge = [
        CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_5x5
        , CONFIG.FILTERS.PIXEL_DIFF_5x5
        , CONFIG.FILTERS.SOBEL_DILATED_5x5
        , CONFIG.FILTERS.PREWITT_DILATED_5x5
        , CONFIG.FILTERS.KIRSCH_DILATED_5x5
        , CONFIG.FILTERS.KITCHEN_MALIN_DILATED_5x5
        , CONFIG.FILTERS.KAYYALI_DILATED_5x5
        , CONFIG.FILTERS.SCHARR_DILATED_5x5
        , CONFIG.FILTERS.KROON_DILATED_5x5
        , CONFIG.FILTERS.ORHEI_DILATED_5x5
    ]

    threshold = 50
    sigma = 2.25

    for edge in first_order_edge:
        ########################################################################################################################
        # First order edge detection magnitude
        ########################################################################################################################
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=sigma)
        edge_result = Application.do_first_order_derivative_operators(port_input_name=blured_img, operator=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=threshold,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')


    first_order_edge = [
        CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_7x7
        , CONFIG.FILTERS.PIXEL_DIFF_7x7
        , CONFIG.FILTERS.SOBEL_DILATED_7x7
        , CONFIG.FILTERS.PREWITT_DILATED_7x7
        , CONFIG.FILTERS.KIRSCH_DILATED_7x7
        , CONFIG.FILTERS.KITCHEN_MALIN_DILATED_7x7
        , CONFIG.FILTERS.KAYYALI_DILATED_7x7
        , CONFIG.FILTERS.SCHARR_DILATED_7x7
        , CONFIG.FILTERS.KROON_DILATED_7x7
        , CONFIG.FILTERS.ORHEI_DILATED_7x7
    ]

    threshold = 50
    sigma = 2.00

    for edge in first_order_edge:
        ########################################################################################################################
        # First order edge detection magnitude
        ########################################################################################################################
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=sigma)
        edge_result = Application.do_first_order_derivative_operators(port_input_name=blured_img, operator=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=threshold,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')


    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='first_order_results',
                                 list_of_data=list_to_save, number_of_series=50,
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_BLURED_L0',
                                 replace_list=[('SEPARATED_PIXEL_DIFFERENCE_', 'Separated Px Dif '),
                                               ('PIXEL_DIFFERENCE_', 'Pixel Dif '),
                                               ('PREWITT_', 'Prewitt '), ('KIRSCH_', 'Kirsch '), ('SOBEL_', 'Sobel '),
                                               ('SCHARR_', 'Scharr '), ('KROON_', 'Kroon '), ('ORHEI_V1_', 'Orhei '), ('ORHEI_', 'Orhei '),
                                               ('KITCHEN_', 'Kitchen '), ('KAYYALI_', 'Kayyali '),
                                               ('DILATED_', 'dilated ')],
                                 inputs=[''], self_contained_list=True, set_legend_left=True,
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='first_order_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7'],
                                 prefix_data_name='FINAL', suffix_data_name='BLURED', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7'],
                                 data_per_variant=['R', 'P', 'F1'], version_separation='DILATED')

    Utils.close_files()


# def main_find_thr_compass_first_order_edges(dataset):
#     Application.delete_folder_appl_out()
#     Benchmarking.delete_folder_benchmark_out()
#
#     Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)
#
#     list_to_save = []
#
#     Application.do_get_image_job(port_output_name='RAW')
#     Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')
#
#     # find best threshold for first level
#     for thr in range(15, 265, 10):
#         print('thr=', thr)
#         edge_result = Application.do_compass_edge_job(port_input_name='GREY', operator=CONFIG.FILTERS.ROBINSON_CROSS_3x3)
#
#         thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
#                                                              input_threshold_type='cv2.THRESH_BINARY',
#                                                              port_output_name='THR_' + str(thr) + '_' + edge_result)
#         thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
#                                                                           port_output_name='FINAL_' + thr_edge_result)
#         list_to_save.append(thin_thr_edge_result + '_L0')
#
#     Application.create_config_file()
#     Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
#     # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
#     Application.run_application()
#
#     # Do bsds benchmarking
#     # Be ware not to activate job_name_in_port in Application.configure_save_pictures
#     Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
#                                                 gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
#                                                 raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
#                                                 jobs_set=list_to_save, do_thinning=False)
#
#     Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='compass_first_order_thr_results_finder',
#                                  list_of_data=list_to_save, number_of_series=30,
#                                  inputs=[''], self_contained_list=True,
#                                  replace_list=[('_ROBINSON_CROSS_3x3', ''), ('THR_', 'Thr=')],
#                                  prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
#                                  save_plot=True, show_plot=False)
#
#     Utils.close_files()


def main_find_sigma_compass_first_order_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    threshold = 65

    # find best threshold for first level
    for sigma in range(25, 500, 25):
        s = sigma / 100
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))
        edge_result = Application.do_compass_edge_job(port_input_name=blured_img, operator=CONFIG.FILTERS.ROBINSON_CROSS_3x3,
                                                      )
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=threshold,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='compass_first_order_sigma_results_finder',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('ROBINSON_CROSS_3x3_BLURED_SIGMA_', 'S='), ('_', '.')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_L0',
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_find_thr_sig_compass_first_order_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    edges = [
        CONFIG.FILTERS.ROBINSON_CROSS_3x3
        , CONFIG.FILTERS.ROBINSON_CROSS_DILATED_5x5
        , CONFIG.FILTERS.ROBINSON_CROSS_DILATED_7x7
    ]

    # find best threshold for first level
    for edge in edges:
        for thr in range(30, 100, 10):
            for sigma in range(200, 350, 25):
                s = sigma / 100
                blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                                    port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))

                edge_result = Application.do_compass_edge_job(port_input_name=blured_img, operator=edge)

                thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                     input_threshold_type='cv2.THRESH_BINARY',
                                                                     port_output_name='THR_' + str(thr) + '_' + edge_result)

                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                                  port_output_name='FINAL_' + thr_edge_result)

                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='compass_first_order_thr_sigma_results_finder',
                                 list_of_data=list_to_save, number_of_series=25,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('_ROBINSON_CROSS_3x3', ''), ('THR_', 'Thr='), ('_BLURED_SIGMA_', ' S='), ('_', '.')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_L0',
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_first_order_compass_edge_detection(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []
    threshold = 50
    sigma = 2.0

    compass_filters = [
        CONFIG.FILTERS.ROBINSON_CROSS_DILATED_7x7
        , CONFIG.FILTERS.ROBINSON_MODIFIED_CROSS_7x7
        , CONFIG.FILTERS.KIRSCH_DILATED_7x7
        , CONFIG.FILTERS.PREWITT_CROSS_DILATED_7x7
    ]

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=sigma, port_output_name='BLURED')

    for edge in compass_filters:
        edge_results = Application.do_compass_edge_job(port_input_name=blured_img, operator=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_results, input_value=threshold,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_results)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_results)
        list_to_save.append(thin_thr_edge_result + '_L0')

    threshold = 50
    sigma = 2.5

    compass_filters = [
        CONFIG.FILTERS.ROBINSON_CROSS_3x3
        , CONFIG.FILTERS.ROBINSON_CROSS_DILATED_5x5
        , CONFIG.FILTERS.ROBINSON_MODIFIED_CROSS_3x3
        , CONFIG.FILTERS.ROBINSON_MODIFIED_CROSS_5x5
        , CONFIG.FILTERS.KIRSCH_CROSS_3x3
        , CONFIG.FILTERS.KIRSCH_DILATED_5x5
        , CONFIG.FILTERS.PREWITT_CROSS_3x3
        , CONFIG.FILTERS.PREWITT_CROSS_DILATED_5x5
    ]

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=sigma, port_output_name='BLURED')

    for edge in compass_filters:
        edge_results = Application.do_compass_edge_job(port_input_name=blured_img, operator=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_results, input_value=threshold,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_results)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_results)
        list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='compass_first_order_results',
                                 list_of_data=list_to_save, number_of_series=50,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('ROBINSON_CROSS_', 'Robinson Cross '), ('KIRSCH_', 'Kirsch Cross '),
                                               ('ROBINSON_MODIFIED_CROSS_', 'Robinson Mod Cross '),
                                               ('PREWITT_COMPASS_', 'Prewitt Compass '),
                                               ('DILATED_', 'Dilated ')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_BLURED_L0',
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='compass_first_order_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', 'Dilated 5x5', 'Dilated 7x7'],
                                 prefix_data_name='FINAL', suffix_data_name='BLURED', level_data_name='L0',
                                 version_data_name=['3x3', 'DILATED_5x5', 'DILATED_7x7'], version_separation='DILATED',
                                 data_per_variant=['R', 'P', 'F1']
                                 )

    Utils.close_files()


# def main_find_thr_frei_chen_edges(dataset):
#     Application.delete_folder_appl_out()
#     Benchmarking.delete_folder_benchmark_out()
#
#     Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)
#
#     list_to_save = []
#
#     Application.do_get_image_job(port_output_name='RAW')
#     Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')
#
#     # find best threshold for first level
#     for thr in range(15, 265, 10):
#         edge_frei, line_frei = Application.do_frei_chen_edge_job(port_input_name='GREY', dilated_kernel=0)
#         thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_frei, input_value=thr,
#                                                              input_threshold_type='cv2.THRESH_BINARY',
#                                                              port_output_name='THR_' + str(thr) + '_' + edge_frei)
#
#         thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
#                                                                           port_output_name='FINAL_' + thr_edge_result)
#         list_to_save.append(thin_thr_edge_result + '_L0')
#
#     Application.create_config_file()
#     Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
#     # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
#     Application.run_application()
#
#     # Do bsds benchmarking
#     # Be ware not to activate job_name_in_port in Application.configure_save_pictures
#     Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
#                                                 gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
#                                                 raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
#                                                 jobs_set=list_to_save, do_thinning=False)
#
#     Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='frei_chen_thr_results_finder',
#                                  list_of_data=list_to_save, number_of_series=30,
#                                  inputs=[''], self_contained_list=True,
#                                  replace_list=[('_FREI_CHEN_EDGE_', ''), ('THR_', 'Thr='), ('3x3', '')],
#                                  prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
#                                  save_plot=True, show_plot=False)
#
#     Utils.close_files()


# def main_find_sigma_frei_edges(dataset):
#     Application.delete_folder_appl_out()
#     Benchmarking.delete_folder_benchmark_out()
#
#     Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)
#
#     list_to_save = []
#
#     Application.do_get_image_job(port_output_name='RAW')
#     Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')
#
#     thr = 105
#
#     # find best threshold for first level
#     for sigma in range(25, 500, 25):
#         s = sigma / 100
#         blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
#                                                             port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))
#         edge_frei, line_frei = Application.do_frei_chen_edge_job(port_input_name=blured_img, dilated_kernel=0)
#         thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_frei, input_value=thr,
#                                                              input_threshold_type='cv2.THRESH_BINARY',
#                                                              port_output_name='THR_' + str(thr) + '_' + edge_frei)
#
#         thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
#                                                                           port_output_name='FINAL_' + thr_edge_result)
#         list_to_save.append(thin_thr_edge_result + '_L0')
#
#     Application.create_config_file()
#     Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
#     # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
#     Application.run_application()
#
#     # Do bsds benchmarking
#     # Be ware not to activate job_name_in_port in Application.configure_save_pictures
#     Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
#                                                 gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
#                                                 raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
#                                                 jobs_set=list_to_save, do_thinning=False)
#
#     Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='frei_edge_sigma_finder',
#                                  list_of_data=list_to_save, number_of_series=30,
#                                  replace_list=[('FREI_CHEN_EDGE_3x3_BLURED_SIGMA_', 'S='), ('_', '.')],
#                                  prefix_to_cut_legend='FINAL_THR_105_', suffix_to_cut_legend='_L0',
#                                  inputs=[''], self_contained_list=True,
#                                  save_plot=True, show_plot=False)
#
#     Utils.close_files()


def main_find_thr_sigma_frei_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')



    # find best threshold for first level
    for dilatation in range(3):
        for thr in range(50, 90, 10):
            for sigma in range(150, 275, 25):
                s = sigma / 100
                blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                                    port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))
                edge_frei, line_frei = Application.do_frei_chen_edge_job(port_input_name=blured_img, dilated_kernel=dilatation)
                thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_frei, input_value=thr,
                                                                     input_threshold_type='cv2.THRESH_BINARY',
                                                                     port_output_name='THR_' + str(thr) + '_' + edge_frei)

                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                                  port_output_name='FINAL_' + thr_edge_result)
                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='frei_edge_sigma_thr_finder',
                                 list_of_data=list_to_save, number_of_series=25,
                                 replace_list=[('FREI_CHEN_EDGE_3x3_BLURED_SIGMA_', 'S='), ('_', '.')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_L0',
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_frei_edges(dataset):
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    thr = 50
    s = 2.5

    # find best threshold for first level
    blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                        port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))

    for dilatation in range(3):
        edge_frei, line_frei = Application.do_frei_chen_edge_job(port_input_name=blured_img, dilated_kernel=dilatation)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_frei, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + edge_frei)
        thr_line_result = Application.do_image_threshold_job(port_input_name=line_frei, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + line_frei)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + edge_frei)
        thin_thr_line_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_line_result,
                                                                          port_output_name='FINAL_' + line_frei)
        list_to_save.append(thin_thr_edge_result + '_L0')
        list_to_save.append(thin_thr_line_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
    #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
    #                                             raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
    #                                             jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='frei_edge_results',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('FREI_CHEN_EDGE_', 'Frei-Chen Edge '), ('FREI_CHEN_LINE_', 'Frei-Chen Line '),
                                               ('_BLURED_SIGMA_2_5_L0', ''),
                                               ('DILATED_', 'Dilated ')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_BLURED_SIGMA_0_075_L0',
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='frei_edge_latex_table_results', print_to_console=True,
                                 list_of_series=['FREI_CHEN_EDGE', 'FREI_CHEN_LINE'],
                                 header_list=['Variant', '', '3x3', 'Dilated 5x5', 'Dilated 7x7'],
                                 prefix_data_name='FINAL', suffix_data_name='BLURED', level_data_name='L0',
                                 version_data_name=['3x3', 'DILATED_5x5', 'DILATED_7x7'],
                                 data_per_variant=['R', 'P', 'F1']
                                 )

    Utils.close_files()


def main_find_thr_laplace_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1]

    # find best threshold for first level
    for edge in laplace_edges:
        for thr in range(45, 100, 10):
            edge_result = Application.do_laplace_job(port_input_name='GREY', kernel=edge)
            thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                 input_threshold_type='cv2.THRESH_BINARY',
                                                                 port_output_name='THR_' + str(thr) + '_' + edge_result)
            thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                              port_output_name='FINAL_' + thr_edge_result)

            list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    # Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    # Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
    #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
    #                                             raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
    #                                             jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='laplace_thr_results_finder',
                                 list_of_data=list_to_save, number_of_series=25,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('THR_', 'Thr='), ('_LAPLACE_V1_3x3', '')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_laplace_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
         CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_2
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_2, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_2
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_3, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_3
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_4, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_4
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_5, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_5
    ]
    thr = 95
    for edge in laplace_edges:
        edge_result = Application.do_laplace_job(port_input_name='GREY', kernel=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + str(thr) + '_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + thr_edge_result)

        list_to_save.append(thin_thr_edge_result + '_L0')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_3
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_4
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5
    ]
    thr = 75
    for edge in laplace_edges:
        edge_result = Application.do_laplace_job(port_input_name='GREY', kernel=edge)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + str(thr) + '_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + thr_edge_result)

        list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    # Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='laplace_edge_results',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('THR_75_LAPLACE_', ''), ('_DILATED_', ' Dilated '),
                                               ('_3x3', ' 3x3'), ('_5x5', ' 5x5')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='laplace_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', 'Dilated 7x7'],
                                 list_of_series=['LAPLACE_V1', 'LAPLACE_V2', 'LAPLACE_V3', 'LAPLACE_V4', 'LAPLACE_V5'],
                                 prefix_data_name='FINAL', suffix_data_name='GREY', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', 'DILATED_7x7'], version_separation='DILATED',
                                 data_per_variant=['R', 'P', 'F1']
                                 )

    Utils.close_files()


def main_find_sigma_log_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1]

    # find best threshold for first level
    for edge in laplace_edges:
        for sigma in range(100, 200, 20):
            s = sigma / 100
            for thr in range(5, 40, 5):
                edge_result = Application.do_log_job(port_input_name='GREY', gaussian_sigma=s,
                                                     laplacian_kernel=edge,
                                                     port_output_name='LOG_' + edge +'_S_' + str(s).replace('.', '_') + '_GREY')
                thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                     input_threshold_type='cv2.THRESH_BINARY',
                                                                     port_output_name='THR_' + str(thr) + '_' + edge_result)
                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                                  port_output_name='FINAL_' + thr_edge_result)

                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='log_thr_results_finder',
                                 list_of_data=list_to_save, number_of_series=25,
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 replace_list=[('THR_', 'Thr='), ('_LOG_LAPLACE_V1_3x3_S_', ' S='), ('_', '.')],
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_log_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_3,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_4,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5
    ]
    thr = 5
    s = 1.80
    for edge in laplace_edges:

        edge_result = Application.do_log_job(port_input_name='GREY', gaussian_sigma=s, laplacian_kernel=edge,
                                             port_output_name='LOG_' + edge + '_S_' + str(s).replace('.', '_') + '_GREY')
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + str(thr) + '_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + thr_edge_result)

        list_to_save.append(thin_thr_edge_result + '_L0')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_2
    ]
    thr = 5
    s = 1.40
    for edge in laplace_edges:

        edge_result = Application.do_log_job(port_input_name='GREY', gaussian_sigma=s, laplacian_kernel=edge,
                                             port_output_name='LOG_' + edge + '_S_' + str(s).replace('.', '_') + '_GREY')
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + str(thr) + '_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + thr_edge_result)

        list_to_save.append(thin_thr_edge_result + '_L0')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_2,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_3,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_4,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_5
    ]
    thr = 15
    s = 1.80
    for edge in laplace_edges:

        edge_result = Application.do_log_job(port_input_name='GREY', gaussian_sigma=s, laplacian_kernel=edge,
                                             port_output_name='LOG_' + edge + '_S_' + str(s).replace('.', '_') + '_GREY')
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + str(thr) + '_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + thr_edge_result)

        list_to_save.append(thin_thr_edge_result + '_L0')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_2,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_3,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_4,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_5
    ]
    thr = 30
    s = 1.80
    for edge in laplace_edges:

        edge_result = Application.do_log_job(port_input_name='GREY', gaussian_sigma=s, laplacian_kernel=edge,
                                             port_output_name='LOG_' + edge + '_S_' + str(s).replace('.', '_') + '_GREY')
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + str(thr) + '_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + thr_edge_result)

        list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    # Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='log_edge_results',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('THR_5_LOG_LAPLACE_', ''), ('_DILATED_', ' Dilated '),
                                               ('_3x3', ' 3x3'), ('_5x5', '  5x5')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_S_1_8_GREY_L0',
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='log_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', 'Dilated 7x7'],
                                 list_of_series=['LAPLACE_V1', 'LAPLACE_V2', 'LAPLACE_V3', 'LAPLACE_V4', 'LAPLACE_V5'],
                                 prefix_data_name='FINAL', suffix_data_name='GREY', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', 'DILATED_7x7'], version_separation='DILATED',
                                 data_per_variant=['R', 'P', 'F1']
                                 )

    Utils.close_files()


def main_find_sigma_marr_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1]

    for edge in laplace_edges:
        for sigma in range(160, 220, 20):
            s = sigma / 100
            for thr in range(20, 50, 10):
                t = thr / 100
                edge_result = Application.do_marr_hildreth_job(port_input_name='GREY', gaussian_sigma=s,
                                                               laplacian_kernel=edge,
                                                               threshold=t)
                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=edge_result,
                                                                                  port_output_name='FINAL_' + edge_result)

                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='mar_sigma_results_finder',
                                 list_of_data=list_to_save, number_of_series=25,
                                 prefix_to_cut_legend='FINAL_MARR_HILDRETH_LAPLACE_V1_3x3', suffix_to_cut_legend='_GREY_L0',
                                 replace_list=[('FINAL_MARR_HILDRETH_LAPLACE', ''),('_S_', 'S='), ('_THR_', ' Thr='), ('_', '.')],
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_marr_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_3
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_4
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5
    ]

    s = 1.8
    t = 0.3

    for edge in laplace_edges:
        edge_result = Application.do_marr_hildreth_job(port_input_name='GREY', gaussian_sigma=s, laplacian_kernel=edge, threshold=t)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')

    laplace_edges = [
       CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
       , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_2
    ]

    s = 1.6
    t = 0.2

    for edge in laplace_edges:
        edge_result = Application.do_marr_hildreth_job(port_input_name='GREY', gaussian_sigma=s, laplacian_kernel=edge, threshold=t)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_2
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_3
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_4
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_5
    ]

    s = 2.0
    t = 0.3

    for edge in laplace_edges:
        edge_result = Application.do_marr_hildreth_job(port_input_name='GREY', gaussian_sigma=s, laplacian_kernel=edge, threshold=t)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_2
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_3
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_4
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_5
    ]

    s = 2.0
    t = 0.2

    for edge in laplace_edges:
        edge_result = Application.do_marr_hildreth_job(port_input_name='GREY', gaussian_sigma=s, laplacian_kernel=edge, threshold=t)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=edge_result,
                                                                          port_output_name='FINAL_' + edge_result)
        list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='marr_edge_results',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 prefix_to_cut_legend='FINAL_MARR_HILDRETH_LAPLACE_',
                                 replace_list=[('_DILATED_', ' Dilated '), ('_3x3', ' 3x3'), ('_5x5', ' 5x5'), ('_GREY_L0', ''),
                                               ('_S_', ' S='), ('_THR_', ' Thr='), ('_', '.')],
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='marr_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', 'Dilated 7x7'],
                                 list_of_series=['LAPLACE_V1', 'LAPLACE_V2', 'LAPLACE_V3', 'LAPLACE_V4', 'LAPLACE_V5'],
                                 prefix_data_name='FINAL', suffix_data_name='GREY', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', 'DILATED_7x7'], version_separation='DILATED',
                                 data_per_variant=['R', 'P', 'F1']
                                 )

    Utils.close_files()


# def main_sigma_finder_canny(dataset):
#     Application.delete_folder_appl_out()
#     Benchmarking.delete_folder_benchmark_out()
#
#     Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)
#
#     list_to_save = []
#
#     Application.do_get_image_job(port_output_name='RAW')
#     Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')
#
#     edge = CONFIG.FILTERS.SOBEL_3x3
#
#     # find best threshold for first level
#     for sigma in range(25, 500, 25):
#         s = sigma / 100
#         blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
#                                                             port_output_name='BLURED_S_' + str(s).replace('.', '_'))
#         Application.do_max_pixel_image_job(port_input_name=blured_img, port_output_name='MAX_' + blured_img)
#         canny_result = Application.do_canny_ratio_threshold_job(port_input_name=blured_img, edge_detector=edge,
#                                                                 port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_'),
#                                                                 canny_config_value='MAX_' + blured_img, do_blur=False)
#         list_to_save.append(canny_result + '_L0')
#
#     Application.create_config_file()
#     Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
#     # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
#     Application.run_application()
#
#     # Do bsds benchmarking
#     # Be ware not to activate job_name_in_port in Application.configure_save_pictures
#     Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
#                                                 gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
#                                                 raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
#                                                 jobs_set=list_to_save, do_thinning=False)
#
#     Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='canny_sigma_results_finder',
#                                  suffix_to_cut_legend='_L0',
#                                  list_of_data=list_to_save, number_of_series=30,
#                                  replace_list=[('CANNY_SOBEL_3x3', ''), ('_S_', 'S='), ('_L0', ''), ('_', '.')],
#                                  inputs=[''], self_contained_list=True,
#                                  save_plot=True, show_plot=False)
#
#     Utils.close_files()


def main_sigma_finder_canny_2(dataset):
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    edges = [
        CONFIG.FILTERS.SOBEL_3x3,
        CONFIG.FILTERS.SOBEL_5x5,
        CONFIG.FILTERS.SOBEL_7x7,
        CONFIG.FILTERS.SOBEL_DILATED_5x5,
        CONFIG.FILTERS.SOBEL_DILATED_7x7
             ]

    # find best threshold for first level
    for edge in edges:
        for sigma in range(100, 175, 25):
            s = sigma / 100
            blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                                port_output_name='BLURED_S_' + str(s).replace('.', '_'))
            for low in range(70, 150, 10):
                for high in range(90, 200, 10):
                # for high in [90]:
                    if low < high:
                        canny_result = Application.do_canny_config_job(port_input_name=blured_img, edge_detector=edge, canny_config=CONFIG.CANNY_VARIANTS.MANUAL_THRESHOLD,
                                                                       low_manual_threshold = low, high_manual_threshold=high, canny_config_value=None,
                                                                       port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_') + '_L_' + str(low) + '_H_' + str(high),
                                                                       do_blur=False)
                        list_to_save.append(canny_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
    #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
    #                                             raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
    #                                             jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='canny_sigma_results_finder',
                                 suffix_to_cut_legend='_L0',
                                 list_of_data=list_to_save, number_of_series=50, set_legend_left=True,
                                 replace_list=[('CANNY_SOBEL', ''),('_DILATED', ' Dilated '),
                                               ('_3x3', ' 3x3'), ('_5x5', ' 5x5'), ('_7x7', ' 7x7'), ('_S_', ' S='), ('_L_', ' L='), ('_H_', ' H='), ('_L0', ''), ('_', '.')],
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


# def main_canny(dataset):
#     Application.delete_folder_appl_out()
#     Benchmarking.delete_folder_benchmark_out()
#
#     Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)
#
#     list_to_save = []
#
#     Application.do_get_image_job(port_output_name='RAW')
#     Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')
#
#     first_order_edge = [
#         CONFIG.FILTERS.PIXEL_DIFF_3x3, CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_3x3
#         , CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_5x5, CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_7x7
#         , CONFIG.FILTERS.PIXEL_DIFF_5x5, CONFIG.FILTERS.PIXEL_DIFF_7x7
#
#         , CONFIG.FILTERS.SOBEL_3x3, CONFIG.FILTERS.SOBEL_5x5, CONFIG.FILTERS.SOBEL_7x7
#         , CONFIG.FILTERS.SOBEL_DILATED_5x5, CONFIG.FILTERS.SOBEL_DILATED_7x7
#
#         , CONFIG.FILTERS.PREWITT_3x3, CONFIG.FILTERS.PREWITT_5x5, CONFIG.FILTERS.PREWITT_7x7
#         , CONFIG.FILTERS.PREWITT_DILATED_5x5, CONFIG.FILTERS.PREWITT_DILATED_7x7
#
#         , CONFIG.FILTERS.KIRSCH_3x3, CONFIG.FILTERS.KIRSCH_5x5
#         , CONFIG.FILTERS.KIRSCH_DILATED_5x5, CONFIG.FILTERS.KIRSCH_DILATED_7x7
#
#         , CONFIG.FILTERS.KITCHEN_MALIN_3x3
#         , CONFIG.FILTERS.KITCHEN_MALIN_DILATED_5x5, CONFIG.FILTERS.KITCHEN_MALIN_DILATED_7x7
#
#         , CONFIG.FILTERS.KAYYALI_3x3
#         , CONFIG.FILTERS.KAYYALI_DILATED_5x5, CONFIG.FILTERS.KAYYALI_DILATED_7x7
#
#         , CONFIG.FILTERS.SCHARR_3x3, CONFIG.FILTERS.SCHARR_5x5
#         , CONFIG.FILTERS.SCHARR_DILATED_5x5, CONFIG.FILTERS.SCHARR_DILATED_7x7
#
#         , CONFIG.FILTERS.KROON_3x3
#         , CONFIG.FILTERS.KROON_DILATED_5x5, CONFIG.FILTERS.KROON_DILATED_7x7
#
#         , CONFIG.FILTERS.ORHEI_3x3, CONFIG.FILTERS.ORHEI_B_5x5
#         , CONFIG.FILTERS.ORHEI_DILATED_5x5, CONFIG.FILTERS.ORHEI_DILATED_7x7
#     ]
#
#     s = 1.25
#     # find best threshold for first level
#     for edge in first_order_edge:
#         blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
#                                                             port_output_name='BLURED_S_' + str(s).replace('.', '_'))
#         Application.do_max_pixel_image_job(port_input_name=blured_img, port_output_name='MAX_' + blured_img)
#         canny_result = Application.do_canny_ratio_threshold_job(port_input_name=blured_img, edge_detector=edge,
#                                                                 port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_'),
#                                                                 canny_config_value='MAX_' + blured_img, do_blur=False)
#         list_to_save.append(canny_result + '_L0')
#
#     Application.create_config_file()
#     Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
#     # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
#     Application.run_application()
#
#     # Do bsds benchmarking
#     # Be ware not to activate job_name_in_port in Application.configure_save_pictures
#     Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
#                                                 gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
#                                                 raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
#                                                 jobs_set=list_to_save, do_thinning=False)
#
#     Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='canny_results',
#                                  suffix_to_cut_legend='_S_1_25_L0', prefix_to_cut_legend='CANNY_',
#                                  list_of_data=list_to_save, number_of_series=40,
#                                  replace_list=[('SEPARATED_PIXEL_DIFFERENCE_', 'Separated Px Dif '),
#                                                ('PIXEL_DIFFERENCE_', 'Pixel Dif '),
#                                                ('PREWITT_', 'Prewitt '), ('KIRSCH_', 'Kirsch '), ('SOBEL_', 'Sobel '),
#                                                ('SCHARR_', 'Scharr '), ('KROON_', 'Kroon '), ('ORHEI_V1_', 'Orhei '),
#                                                ('ORHEI_', 'Orhei '),
#                                                ('KITCHEN_', 'Kitchen '), ('KAYYALI_', 'Kayyali '),
#                                                ('DILATED_', 'dilated ')],
#                                  inputs=[''], self_contained_list=True,
#                                  save_plot=True, show_plot=False)
#
#     Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='canny_latex_table_results', print_to_console=True,
#                                  header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7'],
#                                  prefix_data_name='FINAL', suffix_data_name='BLURED', level_data_name='L0',
#                                  version_data_name=['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7'],
#                                  data_per_variant=['R', 'P', 'F1'], version_separation='DILATED')
#
#     Utils.close_files()


def main_canny_2(dataset):
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    first_order_edge_3x3 = [
        CONFIG.FILTERS.PIXEL_DIFF_3x3, CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_3x3
        , CONFIG.FILTERS.SOBEL_3x3
        , CONFIG.FILTERS.PREWITT_3x3
        , CONFIG.FILTERS.KIRSCH_3x3
        , CONFIG.FILTERS.KITCHEN_MALIN_3x3
        , CONFIG.FILTERS.KAYYALI_3x3
        , CONFIG.FILTERS.SCHARR_3x3
        , CONFIG.FILTERS.KROON_3x3
        , CONFIG.FILTERS.ORHEI_3x3
    ]

    s = 1.5
    # find best threshold for first level
    for edge in first_order_edge_3x3:
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_S_' + str(s).replace('.', '_'))
        low = 80
        high = 90
        canny_result = Application.do_canny_config_job(port_input_name=blured_img, edge_detector=edge, canny_config=CONFIG.CANNY_VARIANTS.MANUAL_THRESHOLD,
                                                       low_manual_threshold=low, high_manual_threshold=high, canny_config_value=None,
                                                       port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_') + '_L_' + str(low) + '_H_' + str(high),
                                                       do_blur=False)
        list_to_save.append(canny_result + '_L0')

    first_order_edge_7x7 = [
        CONFIG.FILTERS.SOBEL_7x7,
        CONFIG.FILTERS.PREWITT_7x7,

    ]

    # find best threshold for first level
    for edge in first_order_edge_7x7:
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_S_' + str(s).replace('.', '_'))
        low = 70
        high = 90
        canny_result = Application.do_canny_config_job(port_input_name=blured_img, edge_detector=edge, canny_config=CONFIG.CANNY_VARIANTS.MANUAL_THRESHOLD,
                                                       low_manual_threshold=low, high_manual_threshold=high, canny_config_value=None,
                                                       port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_') + '_L_' + str(low) + '_H_' + str(high),
                                                       do_blur=False)
        list_to_save.append(canny_result + '_L0')

    first_order_edge = [
         CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_5x5, CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_7x7
        , CONFIG.FILTERS.PIXEL_DIFF_5x5, CONFIG.FILTERS.PIXEL_DIFF_7x7

        , CONFIG.FILTERS.SOBEL_5x5
        , CONFIG.FILTERS.SOBEL_DILATED_5x5, CONFIG.FILTERS.SOBEL_DILATED_7x7

        , CONFIG.FILTERS.PREWITT_5x5
        , CONFIG.FILTERS.PREWITT_DILATED_5x5, CONFIG.FILTERS.PREWITT_DILATED_7x7

        , CONFIG.FILTERS.KIRSCH_5x5
        , CONFIG.FILTERS.KIRSCH_DILATED_5x5, CONFIG.FILTERS.KIRSCH_DILATED_7x7

        , CONFIG.FILTERS.KITCHEN_MALIN_DILATED_5x5, CONFIG.FILTERS.KITCHEN_MALIN_DILATED_7x7

        , CONFIG.FILTERS.KAYYALI_DILATED_5x5, CONFIG.FILTERS.KAYYALI_DILATED_7x7

        , CONFIG.FILTERS.SCHARR_5x5
        , CONFIG.FILTERS.SCHARR_DILATED_5x5, CONFIG.FILTERS.SCHARR_DILATED_7x7

        , CONFIG.FILTERS.KROON_DILATED_5x5, CONFIG.FILTERS.KROON_DILATED_7x7

        , CONFIG.FILTERS.ORHEI_B_5x5
        , CONFIG.FILTERS.ORHEI_DILATED_5x5, CONFIG.FILTERS.ORHEI_DILATED_7x7
    ]

    s = 1.5
    # find best threshold for first level
    for edge in first_order_edge:
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_S_' + str(s).replace('.', '_'))
        low = 90
        high = 130
        canny_result = Application.do_canny_config_job(port_input_name=blured_img, edge_detector=edge, canny_config=CONFIG.CANNY_VARIANTS.MANUAL_THRESHOLD,
                                                       low_manual_threshold = low, high_manual_threshold=high, canny_config_value=None,
                                                       port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_') + '_L_' + str(low) + '_H_' + str(high),
                                                       do_blur=False)
        list_to_save.append(canny_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=True)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='canny_results',
                                 suffix_to_cut_legend=None, prefix_to_cut_legend='CANNY_',
                                 list_of_data=list_to_save, number_of_series=40,
                                 replace_list=[('SEPARATED_PIXEL_DIFFERENCE_', 'Separated Px Dif '),
                                               ('PIXEL_DIFFERENCE_', 'Pixel Dif '),
                                               ('PREWITT_', 'Prewitt '), ('KIRSCH_', 'Kirsch '), ('SOBEL_', 'Sobel '),
                                               ('SCHARR_', 'Scharr '), ('KROON_', 'Kroon '), ('ORHEI_V1_', 'Orhei '),
                                               ('ORHEI_', 'Orhei '),
                                               ('KITCHEN_', 'Kitchen '), ('KAYYALI_', 'Kayyali '),
                                               ('DILATED_', 'dilated '),
                                               ('_3x3', ' 3x3'), ('_5x5', ' 5x5'), ('_7x7', ' 7x7'),
                                               ('_S_', ' S='), ('_L_', ' L='), ('_H_', ' H='), ('_L0', ''), ('_', '.')],
                                 inputs=[''], self_contained_list=True, set_legend_left=True,
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='canny_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7'],
                                 prefix_data_name='CA', suffix_data_name='BLURED', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7'],
                                 data_per_variant=['R', 'P', 'F1'], version_separation='DILATED')

    Utils.close_files()


def main_param_shen_finder(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1
    ]

    for edge in laplace_edges:
        for s in [0.5, 0.9]:
            for w in [5, 11]:
                for r in [0.5, 0.9]:
                    for th in [0, 0.5]:
                        for thr in [4]:
                            edge_result = Application.do_shen_castan_job(port_input_name='GREY',
                                                                         laplacian_kernel=edge,
                                                                         laplacian_threhold=thr, smoothing_factor=s, zc_window_size=w,
                                                                         thinning_factor=th, ratio=r,
                                                                         port_output_name='SHEN_CASTAN_' + edge + '_THR_' + str(thr).replace('.', '_')
                                                                                          + '_S_' + str(s).replace('.', '_') + '_W_' + str(w) +
                                                                                          '_R_' + str(r).replace('.', '_') + '_TH_' + str(
                                                                             th).replace('.', '_'))
                            list_to_save.append(edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
    #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
    #                                             raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
    #                                             jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='', level='L0', order_by='f1', name='shen_tunning',
                                 list_of_data=list_to_save, number_of_series=25,
                                 suffix_to_cut_legend='_L0',
                                 replace_list=[('SHEN_CASTAN_', ''), ('THR_', ' Thr='), ('_S_', ' S='), ('_W_', ' W='),
                                               ('_R_', ' R='), ('_TH_', ' Tn='), ('_', '.')],
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_shen_edges(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_2
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_2, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_2
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_3
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_3, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_3
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_4
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_4, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_4
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_5, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_5
    ]

    thr = 4
    s = 0.5
    w = 11
    th = 0.0
    r = 0.5

    for edge in laplace_edges:
        if 'DILATED_7x7' in edge:
            s = 0.9

        edge_result = Application.do_shen_castan_job(port_input_name='GREY', laplacian_kernel=edge,
                                                     laplacian_threhold=thr, smoothing_factor=s, zc_window_size=w,
                                                     thinning_factor=th, ratio=r,
                                                     port_output_name='SHEN_CASTAN_' + edge)
        list_to_save.append(edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='shen_edge_results',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('SHEN_CASTAN_LAPLACE_', ''), ('_DILATED_', ' Dilated '), ('_3x3', ' 3x3'), ('_5x5', ' 5x5'),
                                               ('_L0', '')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='shen_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', 'Dilated 7x7'],
                                 list_of_series=['LAPLACE_V1', 'LAPLACE_V2', 'LAPLACE_V3', 'LAPLACE_V4', 'LAPLACE_V5'],
                                 prefix_data_name='FINAL', suffix_data_name='GREY', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', 'DILATED_7x7'], version_separation='DILATED',
                                 data_per_variant=['R', 'P', 'F1']
                                 )

    Utils.close_files()


def main_ed_parsing(dataset):
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')

    list = []

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3,
        CONFIG.FILTERS.SOBEL_5x5,
        CONFIG.FILTERS.SOBEL_7x7,
        CONFIG.FILTERS.SOBEL_DILATED_5x5,
        CONFIG.FILTERS.SOBEL_DILATED_7x7
    ]

    for edge in first_order_edge:
        # for kernel_gaus in [3, 5, 7, 9]:
        for kernel_gaus in [5, 7, 9]:
            # for grad_thr in [10,  30, 40, 50, 60, 70, 90, 110, 130, 150]:
            for grad_thr in [40, 50, 60, 70]:
                for anc_thr in [5, 10, 20]:
                    for sc_int in [1]:
                        blur = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', kernel_size=kernel_gaus, sigma=0)
                        e3, e4 = Application.do_edge_drawing_mod_job(port_input_name=blur, operator=edge,
                                                                     gradient_thr=grad_thr, anchor_thr=anc_thr, scan_interval=sc_int,
                                                                     max_edges=100, max_points_edge=100)
                        list.append(e3 + '_L0')


    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list)
    # Application.configure_show_pictures(ports_to_show=list, time_to_show=0)

    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/test',
                                                raw_image='TestData/BSR/BSDS500/data/images/test',
                                                jobs_set=list, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='EDGE_DRAWING_MOD_', level='L0', order_by='f1', name='ed_finder_thr',
                                 list_of_data=list, number_of_series=25,
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 suffix_to_cut_legend='_S_0_GRAY_RAW_L0',
                                 replace_list=[('EDGE_DRAWING_MOD_THR_', 'TG='), ('_ANC_THR_', ' TA='), ('_SCAN_', ' SI='), ('_SOBEL_3x3_GAUSS_BLUR_K_', ' GK=')],
                                 save_plot=True, show_plot=False, set_all_to_legend=False)

    Utils.close_files()


def main_ededge(dataset):
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')
    blur = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', sigma=0, kernel_size=9)

    list_to_eval_edge = []

    first_order_edge = [
         CONFIG.FILTERS.PIXEL_DIFF_3x3, CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_3x3
        , CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_5x5, CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_7x7
        , CONFIG.FILTERS.PIXEL_DIFF_5x5, CONFIG.FILTERS.PIXEL_DIFF_7x7

        , CONFIG.FILTERS.SOBEL_3x3, CONFIG.FILTERS.SOBEL_5x5, CONFIG.FILTERS.SOBEL_7x7
        , CONFIG.FILTERS.SOBEL_DILATED_5x5, CONFIG.FILTERS.SOBEL_DILATED_7x7

        , CONFIG.FILTERS.PREWITT_3x3, CONFIG.FILTERS.PREWITT_5x5, CONFIG.FILTERS.PREWITT_7x7
        , CONFIG.FILTERS.PREWITT_DILATED_5x5, CONFIG.FILTERS.PREWITT_DILATED_7x7

        , CONFIG.FILTERS.KIRSCH_3x3, CONFIG.FILTERS.KIRSCH_5x5
        , CONFIG.FILTERS.KIRSCH_DILATED_5x5, CONFIG.FILTERS.KIRSCH_DILATED_7x7

        , CONFIG.FILTERS.KITCHEN_MALIN_3x3
        , CONFIG.FILTERS.KITCHEN_MALIN_DILATED_5x5, CONFIG.FILTERS.KITCHEN_MALIN_DILATED_7x7

        , CONFIG.FILTERS.KAYYALI_3x3
        , CONFIG.FILTERS.KAYYALI_DILATED_5x5, CONFIG.FILTERS.KAYYALI_DILATED_7x7

        , CONFIG.FILTERS.SCHARR_3x3, CONFIG.FILTERS.SCHARR_5x5
        , CONFIG.FILTERS.SCHARR_DILATED_5x5, CONFIG.FILTERS.SCHARR_DILATED_7x7

        , CONFIG.FILTERS.KROON_3x3
        , CONFIG.FILTERS.KROON_DILATED_5x5, CONFIG.FILTERS.KROON_DILATED_7x7

        , CONFIG.FILTERS.ORHEI_3x3, CONFIG.FILTERS.ORHEI_B_5x5
        , CONFIG.FILTERS.ORHEI_DILATED_5x5, CONFIG.FILTERS.ORHEI_DILATED_7x7
    ]

    for edge in first_order_edge:
        for gr_thr in [50]:
            for anc_thr in [5]:
                e1, e2, = Application.do_edge_drawing_mod_job(port_input_name=blur, operator=edge,
                                                              gradient_thr=gr_thr, anchor_thr=anc_thr, scan_interval=1,
                                                              max_edges=100, max_points_edge=100)
                list_to_eval_edge.append(e1 + '_L0')

    Application.create_config_file(verbose=False)
    Application.configure_save_pictures(job_name_in_port=False, ports_to_save='ALL')
    # Application.configure_show_pictures(ports_to_show=list_to_save, time_to_show=200)

    # Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_eval_edge, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='EDGE_DRAWING_MOD_', level='L0', order_by='f1', name='ed_results',
                                 list_of_data=list_to_eval_edge, number_of_series=50,
                                 inputs=[''], self_contained_list=True, set_legend_left=True,
                                 suffix_to_cut_legend='_S_0_GRAY_RAW_L0',
                                 replace_list=[('EDGE_DRAWING_MOD_THR_50_ANC_THR_5_SCAN_1_', ''),
                                               ('SEPARATED_PIXEL_DIFFERENCE_', 'Separated Px Dif '),
                                               ('PIXEL_DIFFERENCE_', 'Pixel Dif '),
                                               ('PREWITT_', 'Prewitt '), ('KIRSCH_', 'Kirsch '), ('SOBEL_', 'Sobel '),
                                               ('SCHARR_', 'Scharr '), ('KROON_', 'Kroon '), ('ORHEI_V1_', 'Orhei '),
                                               ('ORHEI_', 'Orhei '),
                                               ('KITCHEN_', 'Kitchen '), ('KAYYALI_', 'Kayyali '),
                                               ('DILATED_', 'dilated '),
                                               ('_GAUSS_BLUR_K_9', '')],
                                 save_plot=True, show_plot=False, set_all_to_legend=False)

    # Utils.create_latex_cpm_table_list()

    Utils.close_files()


def main_find_param_first_order_edges_SFOM():
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/dilation_test/test')

    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()
    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')

    list_to_save = list()

    edges = [
        CONFIG.FILTERS.SOBEL_3x3,
        CONFIG.FILTERS.SOBEL_5x5,
        CONFIG.FILTERS.SOBEL_7x7,
        CONFIG.FILTERS.SOBEL_DILATED_5x5,
        CONFIG.FILTERS.SOBEL_DILATED_7x7
             ]

    for edge in edges:
        # find best threshold for first level
        for thr in range(30, 160, 10):
            # for thr in [10]:
            for sigma in range(25, 300, 25):
                # for sigma in [200]:
                s = sigma / 100
                # print('thr=', thr)
                blured_img = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', sigma=s,
                                                                    port_output_name='BLURED_S_' + str(s).replace('.', '_'))
                edge_result = Application.do_first_order_derivative_operators(port_input_name=blured_img,
                                                                              operator=edge)

                thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                     input_threshold_type='cv2.THRESH_BINARY',
                                                                     port_output_name='THR_' + str(thr) + '_' + edge_result)
                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                                  port_output_name='FINAL_' + thr_edge_result)
                list_to_save.append(thin_thr_edge_result + '_L0')


    Application.create_config_file(verbose=False)
    Application.configure_save_pictures(job_name_in_port=False, ports_to_save=list_to_save)

    # Application.run_application()
    #
    # # Do bsds benchmarking
    # # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=1)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=2)
    #
    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,)


    Utils.plot_box_benchmark_values(name_to_save='SFOM_first_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=edges)

    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_first_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_first_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=edges)


def main_ed_parsing_SFOM():
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/dilation_test/test')

    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')

    list = []

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3,
        CONFIG.FILTERS.SOBEL_5x5,
        CONFIG.FILTERS.SOBEL_7x7,
        CONFIG.FILTERS.SOBEL_DILATED_5x5,
        CONFIG.FILTERS.SOBEL_DILATED_7x7
    ]

    for edge in first_order_edge:
        for kernel_gaus in [3, 5, 7, 9]:
        # for kernel_gaus in [9]:
            for grad_thr in [10, 20, 30, 40, 50, 60, 70, 90, 110, 130, 150]:
            # for grad_thr in [10]:
                for anc_thr in [5, 10, 20]:
                # for anc_thr in [20]:
                    for sc_int in [1]:
                        blur = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', kernel_size=kernel_gaus, sigma=0)
                        e3, e4 = Application.do_edge_drawing_mod_job(port_input_name=blur, operator=edge,
                                                                     gradient_thr=grad_thr, anchor_thr=anc_thr, scan_interval=sc_int,
                                                                     max_edges=100, max_points_edge=100)
                        list.append(e3 + '_L0')


    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list)
    # Application.configure_show_pictures(ports_to_show=list, time_to_show=0)

    # Application.run_application()

    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list,)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list,
    #                                k=2)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list,
    #                                k=1)

    Utils.plot_box_benchmark_values(name_to_save='SFOM_ED_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=first_order_edge)

    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_ED_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=first_order_edge)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_ED_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=first_order_edge)

    Utils.close_files()



def main_find_thr_sig_compass_first_order_edges_SFOM():
    Application.set_input_image_folder('TestData/dilation_test/test')

    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    edges = [
        CONFIG.FILTERS.ROBINSON_CROSS_3x3
        , CONFIG.FILTERS.ROBINSON_CROSS_DILATED_5x5
        , CONFIG.FILTERS.ROBINSON_CROSS_DILATED_7x7
    ]

    # find best threshold for first level
    for edge in edges:
        for thr in range(30, 160, 10):
            for sigma in range(25, 350, 25):
                s = sigma / 100
                blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                                    port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))

                edge_result = Application.do_compass_edge_job(port_input_name=blured_img, operator=edge)

                thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                     input_threshold_type='cv2.THRESH_BINARY',
                                                                     port_output_name='THR_' + str(thr) + '_' + edge_result)

                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                                  port_output_name='FINAL_' + thr_edge_result)

                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()
    #
    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                 gt_location='TestData/dilation_test/validate',
    #                                 raw_image='TestData/dilation_test/test',
    #                                 jobs_set=list_to_save, )
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=2)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=1)

    edge_data = ['ROBINSON_CROSS_3x3', 'ROBINSON_CROSS_DILATED_5x5', 'ROBINSON_CROSS_DILATED_7x7']

    Utils.plot_box_benchmark_values(name_to_save='SFOM_compass_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=edge_data)

    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_compass_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=edge_data)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_compass_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=edge_data)

    Utils.close_files()


def main_param_shen_finder_SFOM():
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/dilation_test/test')

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1
    ]

    for edge in laplace_edges:
        for s in [0.5, 0.9]:
            for w in [5, 11]:
                for r in [0.5, 0.9]:
                    for th in [0, 0.5]:
                        for thr in [4]:
                            edge_result = Application.do_shen_castan_job(port_input_name='GREY',
                                                                         laplacian_kernel=edge,
                                                                         laplacian_threhold=thr, smoothing_factor=s, zc_window_size=w,
                                                                         thinning_factor=th, ratio=r,
                                                                         port_output_name='SHEN_CASTAN_' + edge + '_THR_' + str(thr).replace('.', '_')
                                                                                          + '_S_' + str(s).replace('.', '_') + '_W_' + str(w) +
                                                                                          '_R_' + str(r).replace('.', '_') + '_TH_' + str(
                                                                             th).replace('.', '_'))
                            list_to_save.append(edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()
    #
    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                 gt_location='TestData/dilation_test/validate',
    #                                 raw_image='TestData/dilation_test/test',
    #                                 jobs_set=list_to_save, )
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=2)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=1)

    Utils.plot_box_benchmark_values(name_to_save='SFOM_shen_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=laplace_edges)

    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_shen_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=laplace_edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_shen_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=laplace_edges)

    Utils.close_files()


def main_sigma_finder_canny_SFOM():
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/dilation_test/test')

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    edges = [
        CONFIG.FILTERS.SOBEL_3x3,
        CONFIG.FILTERS.SOBEL_5x5,
        CONFIG.FILTERS.SOBEL_7x7,
        CONFIG.FILTERS.SOBEL_DILATED_5x5,
        CONFIG.FILTERS.SOBEL_DILATED_7x7
             ]

    # find best threshold for first level
    for edge in edges:
        for sigma in range(25, 300, 25):
            s = sigma / 100
            blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                                port_output_name='BLURED_S_' + str(s).replace('.', '_'))
            for low in range(70, 150, 10):
                for high in range(90, 200, 10):
                # for high in [90]:
                    if low < high:
                        canny_result = Application.do_canny_config_job(port_input_name=blured_img, edge_detector=edge, canny_config=CONFIG.CANNY_VARIANTS.MANUAL_THRESHOLD,
                                                                       low_manual_threshold = low, high_manual_threshold=high, canny_config_value=None,
                                                                       port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_') + '_L_' + str(low) + '_H_' + str(high),
                                                                       do_blur=False)
                        list_to_save.append(canny_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()
    #
    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                 gt_location='TestData/dilation_test/validate',
    #                                 raw_image='TestData/dilation_test/test',
    #                                 jobs_set=list_to_save, )
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=2)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=1)

    Utils.plot_box_benchmark_values(name_to_save='SFOM_canny_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=edges)

    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_canny_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_canny_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=edges)

    Utils.close_files()


def main_find_sigma_marr_edges_SFOM():
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/dilation_test/test')

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1]

    for edge in laplace_edges:
        for sigma in range(160, 220, 20):
            s = sigma / 100
            for thr in range(20, 50, 10):
                t = thr / 100
                edge_result = Application.do_marr_hildreth_job(port_input_name='GREY', gaussian_sigma=s,
                                                               laplacian_kernel=edge,
                                                               threshold=t)
                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=edge_result,
                                                                                  port_output_name='FINAL_' + edge_result)

                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()
    #
    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                 gt_location='TestData/dilation_test/validate',
    #                                 raw_image='TestData/dilation_test/test',
    #                                 jobs_set=list_to_save, )
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=2)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=1)

    Utils.plot_box_benchmark_values(name_to_save='SFOM_marr_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=laplace_edges)

    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_marr_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=laplace_edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_marr_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=laplace_edges)

    Utils.close_files()


def main_find_sigma_log_edges_SFOM():
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/dilation_test/test')

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1]

    # find best threshold for first level
    for edge in laplace_edges:
        for sigma in range(100, 200, 20):
            s = sigma / 100
            for thr in range(5, 40, 5):
                edge_result = Application.do_log_job(port_input_name='GREY', gaussian_sigma=s,
                                                     laplacian_kernel=edge,
                                                     port_output_name='LOG_' + edge +'_S_' + str(s).replace('.', '_') + '_GREY')
                thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                     input_threshold_type='cv2.THRESH_BINARY',
                                                                     port_output_name='THR_' + str(thr) + '_' + edge_result)
                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                                  port_output_name='FINAL_' + thr_edge_result)

                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()
    #
    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                 gt_location='TestData/dilation_test/validate',
    #                                 raw_image='TestData/dilation_test/test',
    #                                 jobs_set=list_to_save, )
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=2)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=1)

    Utils.plot_box_benchmark_values(name_to_save='SFOM_log_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=laplace_edges)

    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_log_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=laplace_edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_log_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=laplace_edges)

    Utils.close_files()


def main_find_thr_laplace_edges_SFOM():
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/dilation_test/test')

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    laplace_edges = [
        CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1
        , CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1]

    # find best threshold for first level
    for edge in laplace_edges:
        for thr in range(15, 245, 10):
            edge_result = Application.do_laplace_job(port_input_name='GREY', kernel=edge)
            thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                 input_threshold_type='cv2.THRESH_BINARY',
                                                                 port_output_name='THR_' + str(thr) + '_' + edge_result)
            thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                              port_output_name='FINAL_' + thr_edge_result)

            list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    # Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    # Application.run_application()
    #
    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                 gt_location='TestData/dilation_test/validate',
    #                                 raw_image='TestData/dilation_test/test',
    #                                 jobs_set=list_to_save, )
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=2)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=1)

    Utils.plot_box_benchmark_values(name_to_save='SFOM_laplace_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=laplace_edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_laplace_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=laplace_edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_laplace_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=laplace_edges)

    Utils.close_files()

def main_find_thr_sigma_frei_edges_SFOM():
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/dilation_test/test')

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    # find best threshold for first level
    for dilatation in range(3):
        for thr in range(30, 160, 10):
            for sigma in range(25, 320, 25):
                s = sigma / 100
                blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                                    port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))
                edge_frei, line_frei = Application.do_frei_chen_edge_job(port_input_name=blured_img, dilated_kernel=dilatation)
                thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_frei, input_value=thr,
                                                                     input_threshold_type='cv2.THRESH_BINARY',
                                                                     port_output_name='THR_' + str(thr) + '_' + edge_frei)

                thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                                  port_output_name='FINAL_' + thr_edge_result)
                list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.run_application()

    # Benchmarking.run_SFOM_benchmark(input_location='Logs/application_results',
    #                                 gt_location='TestData/dilation_test/validate',
    #                                 raw_image='TestData/dilation_test/test',
    #                                 jobs_set=list_to_save, )
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=2)
    #
    # Benchmarking.run_RDE_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list_to_save,
    #                                k=1)

    edges = ['FREI_CHEN_EDGE_3x3', 'FREI_CHEN_EDGE_DILATED_5x5', 'FREI_CHEN_EDGE_DILATED_7x7']
    Utils.plot_box_benchmark_values(name_to_save='SFOM_frei_tunning', number_decimal=3,
                                    data='SFOM', data_subsets=edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK2_laplace_tunning', number_decimal=3,
    #                                 data='RDEK2', data_subsets=edges)
    #
    # Utils.plot_box_benchmark_values(name_to_save='RDEK1_laplace_tunning', number_decimal=3,
    #                                 data='RDEK1', data_subsets=edges)

    Utils.close_files()


if __name__ == "__main__":
    dataset = 'test'
    # dataset = 'small'

    # main_find_param_first_order_edges(dataset)
    # Utils.reopen_files()
    # main_first_order_edge_detection(dataset)
    # Utils.reopen_files()

    # main_find_thr_sig_compass_first_order_edges(dataset)
    # Utils.reopen_files()
    # main_first_order_compass_edge_detection(dataset)
    # Utils.reopen_files()

    # main_find_thr_sigma_frei_edges(dataset)
    # Utils.reopen_files()
    # main_frei_edges(dataset)
    # Utils.reopen_files()

    main_find_thr_laplace_edges(dataset)
    # Utils.reopen_files()
    # main_laplace_edges(dataset)
    # Utils.reopen_files()

    # main_find_sigma_log_edges(dataset)
    # Utils.reopen_files()
    # main_log_edges(dataset)
    # Utils.reopen_files()

    # main_find_sigma_marr_edges(dataset)
    # Utils.reopen_files()
    # main_marr_edges(dataset)
    # Utils.reopen_files()

    # main_sigma_finder_canny_2(dataset)
    # Utils.reopen_files()
    # main_canny_2(dataset)
    # Utils.reopen_files()

    # main_param_shen_finder(dataset)
    # Utils.reopen_files()
    # main_shen_edges(dataset)
    # Utils.reopen_files()

    # main_ed_parsing(dataset)
    # Utils.reopen_files()
    # main_ededge(dataset)

    # main_find_param_first_order_edges_SFOM()
    # Utils.reopen_files()
    # main_ed_parsing_SFOM()
    # Utils.reopen_files()
    # main_find_thr_sig_compass_first_order_edges_SFOM()
    # Utils.reopen_files()
    # main_param_shen_finder_SFOM()
    # Utils.reopen_files()
    # main_sigma_finder_canny_SFOM()
    # Utils.reopen_files()
    # main_find_sigma_marr_edges_SFOM()
    # Utils.reopen_files()
    # main_find_sigma_log_edges_SFOM()
    # Utils.reopen_files()
    # main_find_thr_laplace_edges_SFOM()
    # main_find_thr_sigma_frei_edges_SFOM()