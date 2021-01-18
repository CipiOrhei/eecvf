import Application
import Benchmarking
import MachineLearning
import config_main as CONFIG
import Utils

"""
Code for paper:
title={End-to-End Computer Vision Framework},
author={Ciprian Orhei, Muguras Mocofan, Silviu Vert, Radu Vasiu},
booktitle={2020 International Symposium on Electronics and Telecommunications (ISETC)},
pages={},
year={2020},
organization={}
"""


def main_isetc():
    # input for ML block
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/train')
    Application.set_output_image_folder('Logs/ML_input')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='train')
    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=['train_L0'])
    Application.run_application()
    # train model using ML block
    MachineLearning.set_input_train_image_folder(location_train='Logs/ML_input/train_L0',
                                                 location_label='TestData/BSR/BSDS500/data/groundTruth/train_img')
    MachineLearning.set_output_model_folder(location_out='Logs/ml_img_results')
    MachineLearning.do_U_net_model(steps_per_epoch=300, epochs=2)
    # run application
    Application.set_input_image_folder('TestData/pattern_full')
    Application.set_output_image_folder('Logs/application_results')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW')
    Application.do_u_net_edge(port_input_name='GRAY_RAW', location_model='unet_edge', port_name_output='EDGE_UNET')
    Application.do_image_threshold_job(port_input_name='EDGE_UNET', input_value=60, input_threshold_type='cv2.THRESH_BINARY',
                                       port_output_name='THR_EDGE_UNET')
    Application.do_thinning_guo_hall_image_job(port_input_name='THR_EDGE_UNET', port_output_name='THINNED_EDGE_UNET')
    Application.do_canny_otsu_median_sigma_job(port_input_name='GRAY_RAW', edge_detector=CONFIG.FILTERS.SOBEL_DILATED_7x7, do_blur=True,
                                               port_output_name='CANNY')
    Application.do_number_edge_pixels(port_input_name='CANNY', port_output_name='NR_EDGE_CANNY')
    Application.do_number_edge_pixels(port_input_name='THINNED_EDGE_UNET', port_output_name='NR_EDGE_THINNED_EDGE_UNET')
    Application.do_edge_label_job(port_input_name='CANNY')
    Application.do_edge_label_job(port_input_name='THINNED_EDGE_UNET')
    Application.create_config_file()
    # Application.configure_save_pictures(ports_to_save=['CANNY_L0', 'THINNED_EDGE_UNET_L0'])
    # Application.configure_save_pictures(ports_to_save='ALL')
    Application.run_application()
    Utils.plot_custom_list(table_number=2, y_plot_name='Edge pixels', port_list=['NR EDGE PX CANNY_L0', 'NR EDGE PX THINNED_EDGE_UNET_L0'],
                           name_to_save='edge_px')
    Utils.plot_custom_list(table_number=2, y_plot_name='Average pixels per edge',
                           port_list=['AVG px/edge CANNY_L0', 'AVG px/edge THINNED_EDGE_UNET_L0'], name_to_save='avg_edge_px')
    Utils.plot_custom_list(table_number=2, y_plot_name='Number of edges', port_list=['Nr Edges CANNY_L0', 'Nr Edges THINNED_EDGE_UNET_L0'],
                           name_to_save='nr_edge')
    # run benchmark
    # Benchmarking.delete_folder_benchmark_out()
    # Benchmarking.run_FOM_benchmark(input_location='Logs/application_results', gt_location='Logs/application_results/CANNY_L0', raw_image='TestData/pattern_mugur_small', jobs_set=['THINNED_EDGE_UNET_L0'])
    # by default a set of statistics are saved in csv file
    Utils.close_files()
    Utils.plot_avg_time_jobs(table_number=2, save_plot=True)


def main():
    # input for ML block
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/train')
    Application.set_output_image_folder('Logs/ML_input')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='train')
    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=['train_L0'])
    Application.run_application()
    # train model using ML block
    MachineLearning.set_input_train_image_folder(
        location_train='Logs/ML_input/train_L0',
        location_label='TestData/BSR/BSDS500/data/groundTruth/train_img')
    MachineLearning.set_output_model_folder(location_out='Logs/ml_img_results')
    MachineLearning.do_U_net_model(steps_per_epoch=300, epochs=2)
    # run application
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/val')
    Application.set_output_image_folder('Logs/application_results')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW')
    Application.do_u_net_edge(port_input_name='GRAY_RAW',
                              location_model='unet_edge',
                              port_name_output='EDGE_UNET')
    Application.do_first_order_derivative_operators(
        port_input_name='GRAY_RAW',
        operator=CONFIG.FILTERS.SOBEL_DILATED_5x5,
        port_output_name='SOBEL')
    Application.do_canny_otsu_median_sigma_job(
        port_input_name='GRAY_RAW',
        edge_detector=CONFIG.FILTERS.SOBEL_DILATED_5x5,
        do_blur=False, port_output_name='CANNY')
    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL')
    Application.run_application()
    # run benchmark
    list_to_evaluate = ['EDGE_UNET_L0', 'SOBEL_L0', 'CANNY_L0']
    Benchmarking.delete_folder_benchmark_out()
    Benchmarking.run_bsds500_boundary_benchmark(
        input_location='Logs/application_results',
        gt_location='TestData/BSR/BSDS500/data/groundTruth/val',
        raw_image='TestData/BSR/BSDS500/data/images/val',
        jobs_set=list_to_evaluate)
    # by default a set of statistics are saved in csv file
    Utils.close_files()
    Utils.plot_avg_time_jobs(table_number=2, save_plot=True)


def main_2():
    # run application
    Application.set_input_image_folder('TestData/pattern_full')
    Application.set_output_image_folder('Logs/application_results')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')
    Application.do_pyramid_level_down_job(port_input_name='RAW', number_of_lvl=2, is_rgb=True,
                                          port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0)
    levels = [CONFIG.PYRAMID_LEVEL.LEVEL_0, CONFIG.PYRAMID_LEVEL.LEVEL_1,
              CONFIG.PYRAMID_LEVEL.LEVEL_2]
    for level in levels:
        Application.do_grayscale_transform_job(port_input_name='RAW', level=level)
        Application.do_dog_job(port_input_name='GRAY_RAW',
                               port_output_name='DoG_GRAY_RAW', level=level,
                               gaussian_kernel_size_1=5, gaussian_sigma_1=1.0,
                               gaussian_kernel_size_2=7, gaussian_sigma_2=1.4)
        Application.do_dog_job(port_input_name='RAW', is_rgb=True,
                               port_output_name='DoG_RAW', level=level,
                               gaussian_kernel_size_1=5, gaussian_sigma_1=1.0,
                               gaussian_kernel_size_2=7, gaussian_sigma_2=1.4)
        Application.do_grayscale_transform_job(port_input_name='DoG_RAW', level=level)
        Application.do_matrix_difference_job(port_input_name_1='DoG_GRAY_RAW',
                                             port_input_name_2='GRAY_DoG_RAW',
                                             port_output_name='ONLY_IN_DoG_GRAY_RAW', level=level)
        Application.do_matrix_difference_job(port_input_name_1='GRAY_DoG_RAW',
                                             port_input_name_2='DoG_GRAY_RAW',
                                             port_output_name='ONLY_IN_GRAY_DoG_RAW', level=level)
        Application.do_number_edge_pixels(port_input_name='ONLY_IN_DoG_GRAY_RAW', level=level)
        Application.do_number_edge_pixels(port_input_name='ONLY_IN_GRAY_DoG_RAW', level=level)
    Application.create_config_file()
    # Application.configure_save_pictures(job_name_in_port=True)
    Application.run_application()
    Utils.close_files()
    Utils.plot_custom_list(port_list=['NR EDGE PX ONLY_IN_GRAY_DoG_RAW_L0',
                                      'NR EDGE PX ONLY_IN_GRAY_DoG_RAW_L1',
                                      'NR EDGE PX ONLY_IN_GRAY_DoG_RAW_L2'],
                           name_to_save='edge_px_only_in_gray_DoG_raw')
    Utils.plot_custom_list(port_list=['NR EDGE PX ONLY_IN_DoG_GRAY_RAW_L0',
                                      'NR EDGE PX ONLY_IN_DoG_GRAY_RAW_L1',
                                      'NR EDGE PX ONLY_IN_DoG_GRAY_RAW_L2'],
                           name_to_save='edge_px_only_in_Dog_gray_raw')
    Utils.plot_time_jobs(port_list=[['Greyscale transform of RAW on L0 W-0',
                                     'Gaussian Blur of GRAY_RAW with K=5 S=1.0 on L0 W-0',
                                     'Gaussian Blur of GRAY_RAW with K=7 S=1.4 on L0 W-0',
                                     'Difference between GAUS_BLUR_K_5_S_1_0_GRAY_RAW W-0 and '
                                     'GAUS_BLUR_K_7_S_1_4_GRAY_RAW W-0 on L0 W-0'],
                                    ['Greyscale transform of RAW on L1 W-0',
                                     'Gaussian Blur of GRAY_RAW with K=5 S=1.0 on L1 W-0',
                                     'Gaussian Blur of GRAY_RAW with K=7 S=1.4 on L1 W-0',
                                     'Difference between GAUS_BLUR_K_5_S_1_0_GRAY_RAW W-0 and '
                                     'GAUS_BLUR_K_7_S_1_4_GRAY_RAW W-0 on L1 W-0'],
                                    ['Greyscale transform of RAW on L2 W-0',
                                     'Gaussian Blur of GRAY_RAW with K=5 S=1.0 on L2 W-0',
                                     'Gaussian Blur of GRAY_RAW with K=7 S=1.4 on L2 W-0',
                                     'Difference between GAUS_BLUR_K_5_S_1_0_GRAY_RAW W-0 and '
                                     'GAUS_BLUR_K_7_S_1_4_GRAY_RAW W-0 on L2 W-0']],
                         series_names=['DoG_GRAY_RAW_L0', 'DoG_GRAY_RAW_L1', 'DoG_GRAY_RAW_L2'],
                         save_plot=True, name_to_save='dog_on_grey_run_time')
    Utils.plot_time_jobs(port_list=[['Gaussian Blur of RAW with K=5 S=1.0 on L0 W-0',
                                     'Gaussian Blur of RAW with K=7 S=1.4 on L0 W-0',
                                     'Difference between GAUS_BLUR_K_5_S_1_0_RAW W-0 and '
                                     'GAUS_BLUR_K_7_S_1_4_RAW W-0 on L0 W-0',
                                     'Greyscale transform of DoG_RAW on L0 W-0'],
                                    ['Gaussian Blur of RAW with K=5 S=1.0 on L1 W-0',
                                     'Gaussian Blur of RAW with K=7 S=1.4 on L1 W-0',
                                     'Difference between GAUS_BLUR_K_5_S_1_0_RAW W-0 and '
                                     'GAUS_BLUR_K_7_S_1_4_RAW W-0 on L1 W-0',
                                     'Greyscale transform of DoG_RAW on L1 W-0'],
                                    ['Gaussian Blur of RAW with K=5 S=1.0 on L2 W-0',
                                     'Gaussian Blur of RAW with K=7 S=1.4 on L2 W-0',
                                     'Difference between GAUS_BLUR_K_5_S_1_0_RAW W-0 and '
                                     'GAUS_BLUR_K_7_S_1_4_RAW W-0 on L2 W-0',
                                     'Greyscale transform of DoG_RAW on L2 W-0']],
                         series_names=['GRAY_DoG_RAW_L0', 'GRAY_DoG_RAW_L1',
                                       'GRAY_DoG_RAW_L2'],
                         save_plot=True, name_to_save='dog_on_color_run_time')


if __name__ == "__main__":
    # main()
    # main_2()
    main_isetc()
