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

"""
This module contains the code used for the following paper:
  title={An analysis of ED line algorithm in urban street-view dataset},
  author={Orhei, Ciprian and Mocofan, Muguras and Vert, Silviu and Vasiu, Radu},
  booktitle={International Conference on Information and Software Technologies},
  pages={123--135},
  year={2021},
  organization={Springer}
"""


def main_sobel_parsing_natural():
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/test')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')

    list = []

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3
    ]

    for edge in first_order_edge:
        for kernel_gaus in [3, 5, 7, 9]:
            for grad_thr in [10,  30, 40, 50, 60, 70, 90, 110, 130, 150]:
                for anc_thr in [10, 20, 30, 40, 60]:
                    for sc_int in [1, 3, 5]:
                        blur = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', kernel_size=kernel_gaus, sigma=0)
                        e3, e4 = Application.do_edge_drawing_mod_job(port_input_name=blur, operator=edge,
                                                                     gradient_thr=grad_thr, anchor_thr=anc_thr, scan_interval=sc_int,
                                                                     max_edges=100, max_points_edge=100)
                        list.append(e3 + '_L0')


    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list)
    # Application.configure_show_pictures(ports_to_show=list, time_to_show=0)

    Application.run_application()

    list.reverse()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/test',
                                                raw_image='TestData/BSR/BSDS500/data/images/test',
                                                jobs_set=list, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='EDGE_DRAWING_MOD_', level='L0', order_by='f1', name='edge_sobel_thr_finding_natural',
                                 list_of_data=list, number_of_series=50,
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 suffix_to_cut_legend='_S_0_GRAY_RAW_L0',
                                 replace_list=[('EDGE_DRAWING_MOD_THR_', 'TG='), ('_ANC_THR_', ' TA='), ('_SCAN_', ' SI='), ('_SOBEL_3x3_GAUSS_BLUR_K_', ' GK=')],
                                 save_plot=True, show_plot=False, set_all_to_legend=True)

    Utils.close_files()


def main_sobel_parsing_urban():
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/TMBuD/img/VAL/png')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')

    list = []

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3
    ]

    for edge in first_order_edge:
        for kernel_gaus in [3, 5, 7, 9]:
            for grad_thr in [10,  30, 40, 50, 60, 70, 90, 110, 130, 150]:
                for anc_thr in [10, 20, 30, 40, 60]:
                    for sc_int in [1, 3, 5]:
                        blur = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', kernel_size=kernel_gaus, sigma=0)
                        e3, e4 = Application.do_edge_drawing_mod_job(port_input_name=blur, operator=edge,
                                                                     gradient_thr=grad_thr, anchor_thr=anc_thr, scan_interval=sc_int,
                                                                     max_edges=100, max_points_edge=100)
                        list.append(e3 + '_L0')


    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list)
    # Application.configure_show_pictures(ports_to_show=list, time_to_show=0)

    Application.run_application()

    list.reverse()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/TMBuD/edge/VAL/mat',
                                                raw_image='TestData/TMBuD/img/VAL/png',
                                                jobs_set=list, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='EDGE_DRAWING_MOD_', level='L0', order_by='f1', name='edge_sobel_thr_finding_urban',
                                 list_of_data=list, number_of_series=50,
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 suffix_to_cut_legend='_S_0_GRAY_RAW_L0',
                                 replace_list=[('EDGE_DRAWING_MOD_THR_', 'TG='), ('_ANC_THR_', ' TA='), ('_SCAN_', ' SI='), ('_SOBEL_3x3_GAUSS_BLUR_K_', ' GK=')],
                                 save_plot=True, show_plot=False, set_all_to_legend=True)

    Utils.close_files()


def main_urban():
    """
    Main function of framework Please look in example_main for all functions you can use
    """
    Application.set_input_image_folder('TestData\TMBuD\img\VAL\png')
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')
    blur = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', sigma=0, kernel_size=9)

    list_to_eval_edge = []
    list_to_eval_line = []

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3
        , CONFIG.FILTERS.PREWITT_3x3
        , CONFIG.FILTERS.KIRSCH_3x3
        , CONFIG.FILTERS.KITCHEN_MALIN_3x3
        , CONFIG.FILTERS.KAYYALI_3x3
        , CONFIG.FILTERS.SCHARR_3x3
        , CONFIG.FILTERS.KROON_3x3
        , CONFIG.FILTERS.ORHEI_3x3
    ]


    for edge in first_order_edge:
        for gr_thr in [30, 40, 50]:
            for anc_thr in [10,20]:
                e1, e2, e3, e4 = Application.do_ed_lines_mod_job(port_input_name=blur, operator=edge, min_line_length=20,
                                                                 gradient_thr=gr_thr, anchor_thr=anc_thr, scan_interval=1, line_fit_err_thr = 1,
                                                                 max_edges=21000, max_points_edge=8000, max_lines=8000, max_points_line=8000

                                                                 )
                list_to_eval_edge.append(e1 + '_L0')
                list_to_eval_line.append(e4 + '_L0')

    Application.create_config_file(verbose=False)
    Application.configure_save_pictures(job_name_in_port=True, ports_to_save='ALL')
    # Application.configure_show_pictures(ports_to_show=list_to_save, time_to_show=200)

    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData\TMBuD\edge\VAL\mat',
                                                raw_image='TestData\TMBuD\img\VAL\png',
                                                jobs_set=list_to_eval_edge, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='EDGE_DRAWING_MOD_', level='L0', order_by='f1', name='edge_urbanl_list',
                                 list_of_data=list_to_eval_edge, number_of_series=50,
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 suffix_to_cut_legend='_S_0_GRAY_RAW_L0',
                                 replace_list=[('EDGE_DRAWING_MOD_THR_', 'TG='), ('_ANC_THR_', ' TA='), ('_SCAN_', ' SI='),
                                               ('_SOBEL_3x3_', ' Sobel'), ('_PREWITT_3x3_', ' Prewitt'), ('_SCHARR_3x3_', ' Scharr'),
                                               ('_KROON_3x3_', ' Kroon'), ('_KITCHEN_3x3_', ' Kitchen'), ('_ORHEI_3x3_', ' Orhei'),
                                               ('_KAYYALI_3x3_', ' Kayyali'),('_KIRSCH_3x3_', ' Kirsch'), ('GAUSS_BLUR_K_9', ''),],
                                 save_plot=True, show_plot=False, set_all_to_legend=True)

    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData\TMBuD\edge\VAL\mat',
                                                raw_image='TestData\TMBuD\img\VAL\png',
                                                jobs_set=list_to_eval_line, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='ED_LINES_IMG_MIN_LEN_20_LINE_FIT_ERR_1_EDGE_DRAWING_MOD_SEGMENTS_', level='L0', order_by='f1', name='line_urban_list',
                                 list_of_data=list_to_eval_line, number_of_series=50,
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 suffix_to_cut_legend='_S_0_GRAY_RAW_L0',
                                 replace_list=[('ED_LINES_IMG_MIN_LEN_20_LINE_FIT_ERR_1_EDGE_DRAWING_MOD_SEGMENTS_', 'TG='), ('_ANC_THR_', ' TA='), ('_SCAN_', ' SI='),
                                               ('_SOBEL_3x3_', ' Sobel'), ('_PREWITT_3x3_', ' Prewitt'), ('_SCHARR_3x3_', ' Scharr'),
                                               ('_KROON_3x3_', ' Kroon'), ('_KITCHEN_3x3_', ' Kitchen'), ('_ORHEI_3x3_', ' Orhei'),
                                               ('_KAYYALI_3x3_', ' Kayyali'),('_KIRSCH_3x3_', ' Kirsch'), ('GAUSS_BLUR_K_9', ''),],
                                 save_plot=True, show_plot=False, set_all_to_legend=True)

    # Utils.create_latex_cpm_table_list()

    Utils.close_files()


def main_natural():
    """
    Main function of framework Please look in example_main for all functions you can use
    """
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/test')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')
    blur = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', sigma=0, kernel_size=9)

    list_to_eval_edge = []
    list_to_eval_line = []

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3
        , CONFIG.FILTERS.PREWITT_3x3
        , CONFIG.FILTERS.KIRSCH_3x3
        , CONFIG.FILTERS.KITCHEN_MALIN_3x3
        , CONFIG.FILTERS.KAYYALI_3x3
        , CONFIG.FILTERS.SCHARR_3x3
        , CONFIG.FILTERS.KROON_3x3
        , CONFIG.FILTERS.ORHEI_3x3
    ]


    for edge in first_order_edge:
        for gr_thr in [30, 40, 50]:
            for anc_thr in [10,20]:
                e1, e2, e3, e4 = Application.do_ed_lines_mod_job(port_input_name=blur, operator=edge, min_line_length=20,
                                                                 gradient_thr=gr_thr, anchor_thr=anc_thr, scan_interval=1, line_fit_err_thr = 1,
                                                                 max_edges=21000, max_points_edge=8000, max_lines=8000, max_points_line=8000

                                                                 )
                list_to_eval_edge.append(e1 + '_L0')
                list_to_eval_line.append(e4 + '_L0')

    Application.create_config_file(verbose=False)
    Application.configure_save_pictures(job_name_in_port=True, ports_to_save='ALL')
    # Application.configure_show_pictures(ports_to_show=list_to_save, time_to_show=200)

    Application.run_application()

    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/test',
                                                raw_image='TestData/BSR/BSDS500/data/images/test',
                                                jobs_set=list_to_eval_edge, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='EDGE_DRAWING_MOD_', level='L0', order_by='f1', name='edge_natural_list',
                                 list_of_data=list_to_eval_edge, number_of_series=50,
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 suffix_to_cut_legend='_S_0_GRAY_RAW_L0',
                                 replace_list=[('EDGE_DRAWING_MOD_THR_', 'TG='), ('_ANC_THR_', ' TA='), ('_SCAN_', ' SI='),
                                               ('_SOBEL_3x3_', ' Sobel'), ('_PREWITT_3x3_', ' Prewitt'), ('_SCHARR_3x3_', ' Scharr'),
                                               ('_KROON_3x3_', ' Kroon'), ('_KITCHEN_3x3_', ' Kitchen'), ('_ORHEI_3x3_', ' Orhei'),
                                               ('_KAYYALI_3x3_', ' Kayyali'),('_KIRSCH_3x3_', ' Kirsch'), ('GAUSS_BLUR_K_9', ''),],
                                 save_plot=True, show_plot=False, set_all_to_legend=True)

    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/test',
                                                raw_image='TestData/BSR/BSDS500/data/images/test',
                                                jobs_set=list_to_eval_line, do_thinning=False)

    Utils.plot_first_cpm_results(prefix='ED_LINES_IMG_MIN_LEN_20_LINE_FIT_ERR_1_EDGE_DRAWING_MOD_SEGMENTS_', level='L0',
                                 order_by='f1', name='line_natural_list',
                                 list_of_data=list_to_eval_line, number_of_series=50,
                                 inputs=[''], self_contained_list=True, set_legend_left=False,
                                 suffix_to_cut_legend='_S_0_GRAY_RAW_L0',
                                 replace_list=[('ED_LINES_IMG_MIN_LEN_20_LINE_FIT_ERR_1_EDGE_DRAWING_MOD_SEGMENTS_', 'TG='), ('_ANC_THR_', ' TA='), ('_SCAN_', ' SI='),
                                               ('_SOBEL_3x3_', ' Sobel'), ('_PREWITT_3x3_', ' Prewitt'), ('_SCHARR_3x3_', ' Scharr'),
                                               ('_KROON_3x3_', ' Kroon'), ('_KITCHEN_3x3_', ' Kitchen'), ('_ORHEI_3x3_', ' Orhei'),
                                               ('_KAYYALI_3x3_', ' Kayyali'),('_KIRSCH_3x3_', ' Kirsch'), ('GAUSS_BLUR_K_9', ''),],
                                 save_plot=True, show_plot=False, set_all_to_legend=True)

    Utils.close_files()


if __name__ == "__main__":
    main_sobel_parsing_natural()
    main_sobel_parsing_urban()
    main_natural()
    main_urban()
