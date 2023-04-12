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

        Application.do_nonlinear_unsharp_filter_job(port_input_name='RAW', is_rgb=True, kernel_size=5, sigma=0, operator=CONFIG.FILTERS.SOBEL_3x3,laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, hpf_strength=0.7)
        Application.do_nonlinear_unsharp_filter_job(port_input_name=grey, is_rgb=False, kernel_size=5, sigma=0, operator=CONFIG.FILTERS.SOBEL_3x3,laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, hpf_strength=0.7)

        for (input, is_rgb) in ([grey, False],):
            # eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1))
            # eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1))
            # eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1))
            # eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1))
            # eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1))
            # eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1))
            # eval_list.append(Application.do_sharpen_filter_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1))

            # strenght = 0.1
            # eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=strenght))
            # eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, strenght=strenght))
            # eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, strenght=strenght))
            # eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, strenght=strenght))
            # eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strenght=strenght))
            # eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, strenght=strenght))
            # eval_list.append(Application.do_unsharp_filter_expanded_job(port_input_name=input, is_rgb=is_rgb, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, strenght=strenght))

            # hpf_strength = 0.7
            # win_size = 7
            # sigma = 35
            # thr = 5

            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            #
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, casacade_version=True,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_5x5_1, casacade_version=True,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_7x7_1, casacade_version=True,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_9x9_1, casacade_version=True,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, casacade_version=True,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1, casacade_version=True,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))
            # eval_list.append(Application.do_constrained_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_9x9_1, casacade_version=True,
            #                                                                         hpf_strenght=hpf_strength, lee_filter_sigma_value=sigma, lee_sigma_filter_window=win_size, threshold_cliping_window=thr))

            for gk in range(10, 100, 10):
                gk = gk /100
                for hpf_strength in range(1, 250, 10):
                    hpf_strength = hpf_strength / 1000
                    eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk,
                                                                                 operator=CONFIG.FILTERS.SOBEL_3x3,
                                                                                 laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                                                                                 hpf_strength=hpf_strength))
                    eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk,
                                                                                 operator=CONFIG.FILTERS.SOBEL_3x3,
                                                                                 laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1,
                                                                                 hpf_strength=hpf_strength))
                    eval_list.append(Application.do_nonlinear_unsharp_filter_job(port_input_name=input, is_rgb=is_rgb, kernel_size=0, sigma=gk,
                                                                                 operator=CONFIG.FILTERS.SOBEL_3x3,
                                                                                 laplace_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1,
                                                                                 hpf_strength=hpf_strength))





        # for el in eval_list:
            # Application.do_histogram_job(port_input_name=el)
            # Application.do_plot_lines_over_columns_job(port_input_name=el, lines=[50], columns=[240, 280])

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

if __name__ == "__main__":
    test_single_line_effect()