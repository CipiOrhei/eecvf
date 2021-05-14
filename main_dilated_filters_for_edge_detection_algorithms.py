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


def main_find_thr_first_order_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    # find best threshold for first level
    for thr in range(15, 255, 10):
        print('thr=', thr)
        edge_result = Application.do_first_order_derivative_operators(port_input_name='GREY', operator=CONFIG.FILTERS.SOBEL_3x3)

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

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='first_order_thr_results_finder',
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_SOBEL_3x3_GREY_L0',
                                 list_of_data=list_to_save, number_of_series=50,
                                 replace_list=[('THR_', 'Thr=')],
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_find_sigma_first_order_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []
    list_to_benchmark = []
    threshold = 65

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    edge = CONFIG.FILTERS.SOBEL_3x3

    # find best threshold for first level
    for sigma in range(25, 500, 25):
        s = sigma / 100
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_S_' + str(s).replace('.', '_'))
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

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='first_order_sigma_results_finder',
                                 prefix_to_cut_legend='FINAL_SOBEL_3x3_', suffix_to_cut_legend='_L0',
                                 list_of_data=list_to_save, number_of_series=50,
                                 replace_list=[('BLURED_S_', 'S='), ('_', '.')],
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_find_param_first_order_edges(dataset):
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

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
                                                                          operator=CONFIG.FILTERS.SOBEL_3x3)

            thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                                 input_threshold_type='cv2.THRESH_BINARY',
                                                                 port_output_name='THR_' + str(thr) + '_' + edge_result)
            thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                              port_output_name='FINAL_' + thr_edge_result)
            list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='first_order_thr_sigma_param_finder',
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_SOBEL_3x3_GREY_L0',
                                 list_of_data=list_to_save, number_of_series=40,
                                 replace_list=[('THR_', 'Thr=')],
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_first_order_edge_detection(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []
    threshold = 50
    sigma = 2.75

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

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
        ########################################################################################################################
        # First order edge detection magnitude
        ########################################################################################################################
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=sigma, port_output_name='BLURED')
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
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='first_order_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7'],
                                 prefix_data_name='FINAL', suffix_data_name='BLURED', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7'],
                                 data_per_variant=['R', 'P', 'F1'], version_separation='DILATED')

    Utils.close_files()


def main_find_thr_compass_first_order_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    # find best threshold for first level
    for thr in range(15, 265, 10):
        print('thr=', thr)
        edge_result = Application.do_compass_edge_job(port_input_name='GREY', operator=CONFIG.FILTERS.ROBINSON_CROSS_3x3)

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

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='compass_first_order_thr_results_finder',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('_ROBINSON_CROSS_3x3', ''), ('THR_', 'Thr=')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 save_plot=True, show_plot=False)

    Utils.close_files()


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
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    # find best threshold for first level
    for thr in range(30, 160, 10):
        for sigma in range(25, 350, 25):
            s = sigma / 100
            blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                                port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))

            edge_result = Application.do_compass_edge_job(port_input_name=blured_img, operator=CONFIG.FILTERS.ROBINSON_CROSS_3x3)

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

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='compass_first_order_thr_sigma_results_finder',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('_ROBINSON_CROSS_3x3', ''), ('THR_', 'Thr=')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_first_order_compass_edge_detection(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []
    threshold = 65
    sigma = 2.00

    compass_filters = [
        CONFIG.FILTERS.ROBINSON_CROSS_3x3
        , CONFIG.FILTERS.ROBINSON_CROSS_DILATED_5x5
        , CONFIG.FILTERS.ROBINSON_CROSS_DILATED_7x7
        , CONFIG.FILTERS.ROBINSON_MODIFIED_CROSS_3x3
        , CONFIG.FILTERS.ROBINSON_MODIFIED_CROSS_5x5
        , CONFIG.FILTERS.ROBINSON_MODIFIED_CROSS_7x7
        , CONFIG.FILTERS.KIRSCH_CROSS_3x3
        , CONFIG.FILTERS.KIRSCH_DILATED_5x5
        , CONFIG.FILTERS.KIRSCH_DILATED_7x7
        , CONFIG.FILTERS.PREWITT_CROSS_3x3
        , CONFIG.FILTERS.PREWITT_CROSS_DILATED_5x5
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


def main_find_thr_frei_chen_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    # find best threshold for first level
    for thr in range(15, 265, 10):
        edge_frei, line_frei = Application.do_frei_chen_edge_job(port_input_name='GREY', dilated_kernel=0)
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

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='frei_chen_thr_results_finder',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('_FREI_CHEN_EDGE_', ''), ('THR_', 'Thr='), ('3x3', '')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_find_sigma_frei_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    thr = 105

    # find best threshold for first level
    for sigma in range(25, 500, 25):
        s = sigma / 100
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_SIGMA_' + str(s).replace('.', '_'))
        edge_frei, line_frei = Application.do_frei_chen_edge_job(port_input_name=blured_img, dilated_kernel=0)
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

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='frei_edge_sigma_finder',
                                 list_of_data=list_to_save, number_of_series=30,
                                 replace_list=[('FREI_CHEN_EDGE_3x3_BLURED_SIGMA_', 'S='), ('_', '.')],
                                 prefix_to_cut_legend='FINAL_THR_105_', suffix_to_cut_legend='_L0',
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_frei_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    thr = 105
    s = 0.75

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
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='frei_edge_results',
                                 list_of_data=list_to_save, number_of_series=30,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('FREI_CHEN_EDGE_', 'Frei-Chen Edge '), ('FREI_CHEN_LINE_', 'Frei-Chen Line '),
                                               ('_BLURED_SIGMA_0_75_L0', ''),
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
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    # find best threshold for first level
    for thr in range(15, 265, 10):
        edge_result = Application.do_laplace_job(port_input_name='GREY', kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1)
        thr_edge_result = Application.do_image_threshold_job(port_input_name=edge_result, input_value=thr,
                                                             input_threshold_type='cv2.THRESH_BINARY',
                                                             port_output_name='THR_' + str(thr) + '_' + edge_result)
        thin_thr_edge_result = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge_result,
                                                                          port_output_name='FINAL_' + thr_edge_result)

        list_to_save.append(thin_thr_edge_result + '_L0')

    Application.create_config_file()
    # Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
                                                raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
                                                jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='laplace_thr_results_finder',
                                 list_of_data=list_to_save, number_of_series=50,
                                 inputs=[''], self_contained_list=True,
                                 replace_list=[('THR_', 'Thr='), ('_LAPLACE_V1_3x3', '')],
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_laplace_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

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
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
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
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    for sigma in range(20, 200, 20):
        s = sigma / 100
        for thr in range(5, 60, 10):
            edge_result = Application.do_log_job(port_input_name='GREY', gaussian_sigma=s,
                                                 laplacian_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                                                 port_output_name='LOG_LAPLACE_V1_3x3_S_' + str(s).replace('.', '_') + '_GREY')
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
                                 list_of_data=list_to_save, number_of_series=30,
                                 prefix_to_cut_legend='FINAL_', suffix_to_cut_legend='_GREY_L0',
                                 replace_list=[('THR_', 'Thr='), ('_LOG_LAPLACE_V1_3x3_S_', ' S='), ('_', '.')],
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_log_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

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

    Application.create_config_file()
    # Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=False)
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
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
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    for sigma in range(20, 300, 20):
        s = sigma / 100
        for thr in range(30, 100, 30):
            t = thr / 100
            edge_result = Application.do_marr_hildreth_job(port_input_name='GREY', gaussian_sigma=s,
                                                           laplacian_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
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
                                 list_of_data=list_to_save, number_of_series=30,
                                 prefix_to_cut_legend='FINAL_MARR_HILDRETH_LAPLACE_V1_3x3', suffix_to_cut_legend='_GREY_L0',
                                 replace_list=[('_S_', 'S='), ('_THR_', ' Thr='), ('_', '.')],
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_marr_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

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

    s = 1.8
    t = 0.3

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


def main_sigma_finder_canny(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    edge = CONFIG.FILTERS.SOBEL_3x3

    # find best threshold for first level
    for sigma in range(25, 500, 25):
        s = sigma / 100
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_S_' + str(s).replace('.', '_'))
        Application.do_max_pixel_image_job(port_input_name=blured_img, port_output_name='MAX_' + blured_img)
        canny_result = Application.do_canny_ratio_threshold_job(port_input_name=blured_img, edge_detector=edge,
                                                                port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_'),
                                                                canny_config_value='MAX_' + blured_img, do_blur=False)
        list_to_save.append(canny_result + '_L0')

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

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by=None, name='canny_sigma_results_finder',
                                 suffix_to_cut_legend='_L0',
                                 list_of_data=list_to_save, number_of_series=30,
                                 replace_list=[('CANNY_SOBEL_3x3', ''), ('_S_', 'S='), ('_L0', ''), ('_', '.')],
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_sigma_finder_canny_2(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    edge = CONFIG.FILTERS.SOBEL_3x3

    # find best threshold for first level
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
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
    #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
    #                                             raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
    #                                             jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='canny_sigma_results_finder',
                                 suffix_to_cut_legend='_L0',
                                 list_of_data=list_to_save, number_of_series=30,
                                 replace_list=[('CANNY_SOBEL_3x3', ''), ('_S_', ' S='), ('_L_', ' L='), ('_H_', ' H='), ('_L0', ''), ('_', '.')],
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_canny(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

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

    s = 1.25
    # find best threshold for first level
    for edge in first_order_edge:
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_S_' + str(s).replace('.', '_'))
        Application.do_max_pixel_image_job(port_input_name=blured_img, port_output_name='MAX_' + blured_img)
        canny_result = Application.do_canny_ratio_threshold_job(port_input_name=blured_img, edge_detector=edge,
                                                                port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_'),
                                                                canny_config_value='MAX_' + blured_img, do_blur=False)
        list_to_save.append(canny_result + '_L0')

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

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='canny_results',
                                 suffix_to_cut_legend='_S_1_25_L0', prefix_to_cut_legend='CANNY_',
                                 list_of_data=list_to_save, number_of_series=40,
                                 replace_list=[('SEPARATED_PIXEL_DIFFERENCE_', 'Separated Px Dif '),
                                               ('PIXEL_DIFFERENCE_', 'Pixel Dif '),
                                               ('PREWITT_', 'Prewitt '), ('KIRSCH_', 'Kirsch '), ('SOBEL_', 'Sobel '),
                                               ('SCHARR_', 'Scharr '), ('KROON_', 'Kroon '), ('ORHEI_V1_', 'Orhei '),
                                               ('ORHEI_', 'Orhei '),
                                               ('KITCHEN_', 'Kitchen '), ('KAYYALI_', 'Kayyali '),
                                               ('DILATED_', 'dilated ')],
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='canny_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7'],
                                 prefix_data_name='FINAL', suffix_data_name='BLURED', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7'],
                                 data_per_variant=['R', 'P', 'F1'], version_separation='DILATED')

    Utils.close_files()


def main_canny_2(dataset):
    Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

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

    s = 1.5
    # find best threshold for first level
    for edge in first_order_edge:
        blured_img = Application.do_gaussian_blur_image_job(port_input_name='GREY', sigma=s,
                                                            port_output_name='BLURED_S_' + str(s).replace('.', '_'))
        low = 80
        high = 90
        canny_result = Application.do_canny_config_job(port_input_name=blured_img, edge_detector=edge, canny_config=CONFIG.CANNY_VARIANTS.MANUAL_THRESHOLD,
                                                       low_manual_threshold = low, high_manual_threshold=high, canny_config_value=None,
                                                       port_output_name='CANNY_' + edge + '_S_' + str(s).replace('.', '_') + '_L_' + str(low) + '_H_' + str(high),
                                                       do_blur=False)
        list_to_save.append(canny_result + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list_to_save, job_name_in_port=True)
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
    #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/' + dataset,
    #                                             raw_image='TestData/BSR/BSDS500/data/images/' + dataset,
    #                                             jobs_set=list_to_save, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='FINAL', level='L0', order_by='f1', name='canny_results',
                                 suffix_to_cut_legend='_S_1_5_L_80_H_90_L0', prefix_to_cut_legend='CANNY_',
                                 list_of_data=list_to_save, number_of_series=40,
                                 replace_list=[('SEPARATED_PIXEL_DIFFERENCE_', 'Separated Px Dif '),
                                               ('PIXEL_DIFFERENCE_', 'Pixel Dif '),
                                               ('PREWITT_', 'Prewitt '), ('KIRSCH_', 'Kirsch '), ('SOBEL_', 'Sobel '),
                                               ('SCHARR_', 'Scharr '), ('KROON_', 'Kroon '), ('ORHEI_V1_', 'Orhei '),
                                               ('ORHEI_', 'Orhei '),
                                               ('KITCHEN_', 'Kitchen '), ('KAYYALI_', 'Kayyali '),
                                               ('DILATED_', 'dilated ')],
                                 inputs=[''], self_contained_list=True,
                                 save_plot=True, show_plot=False)

    Utils.create_latex_cpm_table(list_of_data=list_to_save, name_of_table='canny_latex_table_results', print_to_console=True,
                                 header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7'],
                                 prefix_data_name='CA', suffix_data_name='BLURED', level_data_name='L0',
                                 version_data_name=['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7'],
                                 data_per_variant=['R', 'P', 'F1'], version_separation='DILATED')

    Utils.close_files()


def main_param_shen_finder(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/' + dataset)

    list_to_save = []

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GREY')

    for s in [0.5, 0.9]:
        for w in [5, 7, 11]:
            for r in [0.5, 0.9]:
                for th in [0, 0.5, 0.9]:
                    for thr in [4]:
                        edge_result = Application.do_shen_castan_job(port_input_name='GREY',
                                                                     laplacian_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                                                                     laplacian_threhold=thr, smoothing_factor=s, zc_window_size=w,
                                                                     thinning_factor=th, ratio=r,
                                                                     port_output_name='SHEN_CASTAN_' + 'THR_' + str(thr).replace('.', '_')
                                                                                      + '_S_' + str(s).replace('.', '_') + '_W_' + str(w) +
                                                                                      '_R_' + str(r).replace('.', '_') + '_TH_' + str(
                                                                         th).replace('.', '_'))
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

    Utils.plot_first_cpm_results(prefix='', level='L0', order_by='f1', name='shen_tunning',
                                 list_of_data=list_to_save, number_of_series=42,
                                 suffix_to_cut_legend='_L0',
                                 replace_list=[('SHEN_CASTAN_', ''), ('THR_', ' Thr='), ('_S_', ' S='), ('_W_', ' W='),
                                               ('_R_', ' R='), ('_TH_', ' Tn='), ('_', '.')],
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 save_plot=True, show_plot=False)

    Utils.close_files()


def main_shen_edges(dataset):
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

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
    s = 0.9
    w = 7
    th = 0.5
    r = 0.9

    for edge in laplace_edges:
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


if __name__ == "__main__":
    # dataset = 'test'
    dataset = 'small'
    # main_find_thr_first_order_edges(dataset)
    # Utils.reopen_files()
    # main_find_sigma_first_order_edges(dataset)
    # Utils.reopen_files()
    # main_find_param_first_order_edges(dataset)
    # Utils.reopen_files()
    # main_first_order_edge_detection(dataset)
    # Utils.reopen_files()
    # main_find_thr_compass_first_order_edges(dataset)
    # Utils.reopen_files()
    # main_find_sigma_compass_first_order_edges(dataset)
    # Utils.reopen_files()
    # main_first_order_compass_edge_detection(dataset)
    # Utils.reopen_files()
    # main_find_thr_frei_chen_edges(dataset)
    # Utils.reopen_files()
    # main_find_sigma_frei_edges(dataset)
    # Utils.reopen_files()
    # main_frei_edges(dataset)
    # Utils.reopen_files()
    # main_find_thr_laplace_edges(dataset)
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
    # main_sigma_finder_canny(dataset)
    # main_sigma_finder_canny_2(dataset)
    # Utils.reopen_files()
    # main_canny(dataset)
    main_canny_2(dataset)
    # main_param_shen_finder(dataset)
    # Utils.reopen_files()
    # main_shen_edges(dataset)
    # Utils.create_latex_cpm_table_list(variants=['FINAL', 'CANNY'], variants_public=['Magnitude Gradient', 'Canny'],
    #                                   sub_variants=[['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7'],
    #                                                 ['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7']],
    #                                   sub_variants_pub=[['3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7'],
    #                                                     ['3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7']],
    #                                   operators=['PIXEL_DIFFERENCE', 'SEPARATED_PIXEL_DIFFERENCE', 'SOBEL', 'PREWITT', 'KIRSCH', 'KITCHEN',
    #                                              'KAYYALI', 'SCHARR', 'KROON', 'ORHEI'],
    #                                   operators_pub=['Pixel Diff', 'Separated Pixle Diff', 'Sobel', 'Prewitt', 'Kirsch', 'Kitchen',
    #                                                  'Kayyali', 'Scharr', 'Kroon', 'Orhei'],
    #                                   inputs=['BLURED', 'S_1_25'], levels=['L0', 'L0'],
    #                                   order=['R', 'P', 'F1'], name_of_table='ortho_table')
    #
    # Utils.create_latex_cpm_table_list(variants=['FINAL_THR_75', 'FINAL_THR_5', 'FINAL_MARR_HILDRETH', 'SHEN_CASTAN'],
    #                                   variants_public=['Laplace', 'LoG', 'Marr-Hildreth', 'Shen-Castan'],
    #                                   sub_variants=[['3x3', '5x5', 'DILATED_5x5', 'DILATED_7x7'],
    #                                                 ['3x3', '5x5', 'DILATED_5x5', 'DILATED_7x7'],
    #                                                 ['3x3', '5x5', 'DILATED_5x5', 'DILATED_7x7'],
    #                                                 ['3x3', '5x5', 'DILATED_5x5', 'DILATED_7x7']],
    #                                   sub_variants_pub=[['3x3', '5x5', 'Dilated 5x5', 'Dilated 7x7'],
    #                                                     ['3x3', '5x5', 'Dilated 5x5', 'Dilated 7x7'],
    #                                                     ['3x3', '5x5', 'Dilated 5x5', 'Dilated 7x7'],
    #                                                     ['3x3', '5x5', 'Dilated 5x5', 'Dilated 7x7']],
    #                                   operators=['LAPLACE_V1', 'LAPLACE_V2', 'LAPLACE_V3', 'LAPLACE_V4', 'LAPLACE_V5'],
    #                                   operators_pub=['V1', 'V2', 'V3', 'V4', 'V5'],
    #                                   inputs=['GREY', 'GREY', 'S_1_8_THR_0_3_GREY', ''], levels=['L0', 'L0', 'L0', 'L0', 'L0'],
    #                                   order=['R', 'P', 'F1'], name_of_table='laplace_table')
