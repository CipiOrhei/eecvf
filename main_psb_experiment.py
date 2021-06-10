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

import cv2

def prepare_dataset():
    """
    Please download http://www.cvlibs.net/datasets/kitti/eval_road.php for this experiment
    :return: None
    """
    Application.set_input_image_folder('TestData/data_road/training/gt_image_2')
    Application.set_output_image_folder('Logs/data_road/labels')
    Application.do_get_image_job(port_output_name='RAW')
    #                OTHER      NOT-ROAD        ROAD
    KITTI_LABELS = [(0, 0, 0), (0, 0, 255), (255, 0, 255)]
    KITTI_LABELS_2 = [(0, 0, 0), (0, 0, 0), (255, 0, 255)]
    KITTI_LABELS_CORRELATION = [0, 0, 1]

    Application.do_class_correlation(port_input_name='RAW', port_output_name='NEW_LABELS_PNG', class_list_in=KITTI_LABELS, class_list_out=KITTI_LABELS_2)
    Application.do_class_correlation(port_input_name='RAW', port_output_name='LABELS', class_list_in=KITTI_LABELS, class_list_out=KITTI_LABELS_CORRELATION)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()


def main_training_data(height, width):
    Application.set_output_image_folder('Logs/application_results_ml_raw')
    Application.set_input_image_folder('TestData/data_road/training/image_2')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')
    list_of_ports_to_move = list()
    list_of_ports_to_move.append(Application.do_resize_image_job(port_input_name='RAW', new_height=height, new_width=width, is_rgb=True, interpolation=cv2.INTER_CUBIC, port_output_name='RAW_RESIZE'))
    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    for el in range(len(list_of_ports_to_move.copy())):
        list_of_ports_to_move[el] += '_LC0'
    Application.create_folders_from_list_ports(folder_names=['Logs/ml_exchange/TRAIN_INPUT', 'Logs/ml_results/VAL_INPUT'], list_port=list_of_ports_to_move, folder_ratios=[0.5, 0.5])

    Application.set_input_image_folder('Logs/ml_exchange/TRAIN_INPUT')
    Application.set_output_image_folder('Logs/application_results_ml_raw')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')
    list_of_ports_to_move = []
    list_of_ports_to_move.append(Application.do_resize_image_job(port_input_name='RAW', new_height=height, new_width=width, is_rgb=True, interpolation=cv2.INTER_CUBIC, port_output_name='RAW_RESIZE'))
    list_of_ports_to_move.append(Application.do_flip_image_job(port_input_name='RAW_RESIZE', is_rgb=True, flip_horizontal=True, flip_vertical=False, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_zoom_image_job(port_input_name='RAW_RESIZE', is_rgb=True, zoom_factor=1.05, do_interpolation=True,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_zoom_image_job(port_input_name='RAW_RESIZE', is_rgb=True, zoom_factor=1.1, do_interpolation=True,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_rotate_image_job(port_input_name='RAW_RESIZE', is_rgb=True, angle=10, reshape=False, extend_border=True,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_rotate_image_job(port_input_name='RAW_RESIZE', is_rgb=True, angle=-10, reshape=False, extend_border=True,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_gaussian_blur_noise_job(port_input_name='RAW_RESIZE', is_rgb=True, mean_value=0.01, variance=0.01,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_gaussian_blur_noise_job(port_input_name='RAW_RESIZE', is_rgb=True, mean_value=0.05, variance=0.05,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_salt_pepper_noise(port_input_name='RAW_RESIZE', is_rgb=True, density=0.1, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_salt_pepper_noise(port_input_name='RAW_RESIZE', is_rgb=True, density=0.4, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_gaussian_blur_image_job(port_input_name='RAW_RESIZE', is_rgb=True, sigma=1.4, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_gaussian_blur_image_job(port_input_name='RAW_RESIZE', is_rgb=True, sigma=3, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_mean_blur_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel_size=9, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_mean_blur_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel_size=7, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_motion_blur_filter_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel_size=7, angle=90,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_motion_blur_filter_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel_size=7, angle=-90,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=1.5,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, beta=10,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=1.5, beta=10,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=2.5, beta=10,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=0.5, beta=0, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=True, alpha=0.1, beta=50, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_pixelate_image_job(port_input_name='RAW_RESIZE', is_rgb=True, nr_pixels_to_group=3,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_pixelate_image_job(port_input_name='RAW_RESIZE', is_rgb=True, nr_pixels_to_group=2,level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_sharpen_filter_job(port_input_name='RAW_RESIZE', is_rgb=True, kernel=3, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    for el in range(len(list_of_ports_to_move.copy())):
        list_of_ports_to_move[el] += '_LC0'

    Application.create_folder_from_list_ports(folder_name='Logs/ml_results/TRAIN_INPUT', list_port=list_of_ports_to_move)

    Utils.close_files()

def main_training_label(height, width):
    Application.set_output_image_folder('Logs/application_results_ml_labels')
    Application.set_input_image_folder('Logs/data_road/labels/LABELS_L0')
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW_LABEL')
    Application.do_grayscale_transform_job(port_input_name='RAW_LABEL', port_output_name='GREY_LABEL')
    list_of_ports_to_move = list()

    list_of_ports_to_move.append( Application.do_resize_image_job(port_input_name='GREY_LABEL', new_height=height, new_width=width, is_rgb=False, interpolation=cv2.INTER_NEAREST, port_output_name='RAW_RESIZE'))

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    for el in range(len(list_of_ports_to_move.copy())):
        list_of_ports_to_move[el] += '_LC0'

    Application.create_folders_from_list_ports(folder_names=['Logs/ml_exchange/TRAIN_LABELS', 'Logs/ml_results/VAL_LABEL'],
                                               list_port=list_of_ports_to_move, folder_ratios=[0.5, 0.5])

    Application.set_input_image_folder('Logs/ml_exchange/TRAIN_LABELS')
    Application.set_output_image_folder('Logs/application_results_ml_labels')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW_LABEL')
    Application.do_grayscale_transform_job(port_input_name='RAW_LABEL', port_output_name='GREY_LABEL')
    list_of_ports_to_move = []

    list_of_ports_to_move.append(Application.do_resize_image_job(port_input_name='GREY_LABEL', new_height=height, new_width=width, is_rgb=False, interpolation=cv2.INTER_NEAREST, port_output_name='RAW_RESIZE'))
    list_of_ports_to_move.append(Application.do_flip_image_job(port_input_name='RAW_RESIZE', is_rgb=False, flip_horizontal=True, flip_vertical=False, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_zoom_image_job(port_input_name='RAW_RESIZE', is_rgb=False, zoom_factor=1.05, do_interpolation=False, port_output_name='ZOOM_1.05_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_zoom_image_job(port_input_name='RAW_RESIZE', is_rgb=False, zoom_factor=1.1, do_interpolation=False, port_output_name='ZOOM_1.1_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_rotate_image_job(port_input_name='RAW_RESIZE', is_rgb=False, angle=10, reshape=False, extend_border=True, do_interpolation=False, port_output_name='ROTATE_ANGLE_10_BORDER_EXT_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_rotate_image_job(port_input_name='RAW_RESIZE', is_rgb=False, angle=-10, reshape=False, extend_border=True, do_interpolation=False, port_output_name='ROTATE_ANGLE_-10_BORDER_EXT_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_gaussian_blur_noise_job(port_input_name='RAW_RESIZE', is_rgb=False, mean_value=0, variance=0, port_output_name='GAUSS_NOISE_MEAN_VAL_0_01_VAR_0_01_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_gaussian_blur_noise_job(port_input_name='RAW_RESIZE', is_rgb=False, mean_value=0, variance=0, port_output_name='GAUSS_NOISE_MEAN_VAL_0_05_VAR_0_05_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_salt_pepper_noise(port_input_name='RAW_RESIZE', is_rgb=False, density=0, port_output_name='S&P_NOISE_DENS_0_1_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_add_salt_pepper_noise(port_input_name='RAW_RESIZE', is_rgb=False, density=0, port_output_name='S&P_NOISE_DENS_0_4_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_gaussian_blur_image_job(port_input_name='RAW_RESIZE', is_rgb=False, sigma=0, port_output_name='GAUSS_BLUR_K_0_S_1_4_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_gaussian_blur_image_job(port_input_name='RAW_RESIZE', is_rgb=False, sigma=0, port_output_name='GAUSS_BLUR_K_0_S_3_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_mean_blur_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel_size=0, port_output_name='MEAN_BLUR_K_9_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_mean_blur_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel_size=0, port_output_name='MEAN_BLUR_K_7_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_motion_blur_filter_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel_size=0, angle=0, port_output_name='MOTION_BLUR_K_7_ANGLE_90_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_motion_blur_filter_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel_size=0, angle=0, port_output_name='MOTION_BLUR_K_7_ANGLE_-90_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, port_output_name='CHANGE_ALPHA_1.5_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, beta=0, port_output_name='CHANGE_BETA_10_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, beta=0, port_output_name='CHANGE_ALPHA_1.5_BETA_10_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, beta=0, port_output_name='CHANGE_ALPHA_2.5_BETA_10_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, beta=0, port_output_name='CHANGE_ALPHA_0.5_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_contrast_brightness_change_image_job(port_input_name='RAW_RESIZE', is_rgb=False, alpha=1, beta=0, port_output_name='CHANGE_ALPHA_0.1_BETA_50_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_pixelate_image_job(port_input_name='RAW_RESIZE', is_rgb=False, nr_pixels_to_group=3, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_pixelate_image_job(port_input_name='RAW_RESIZE', is_rgb=False, nr_pixels_to_group=2, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))
    list_of_ports_to_move.append(Application.do_sharpen_filter_job(port_input_name='RAW_RESIZE', is_rgb=False, kernel=0, port_output_name='SHARPEN_K_3_RAW_RESIZE', level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0))

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()

    for el in range(len(list_of_ports_to_move.copy())):
        list_of_ports_to_move[el] += '_LC0'

    Application.create_folder_from_list_ports(folder_name='Logs/ml_results/TRAIN_LABEL', list_port=list_of_ports_to_move)

    Utils.close_files()

def train_model(height, width):
    MachineLearning.set_image_input_folder('Logs/ml_results/TRAIN_INPUT')
    MachineLearning.set_label_input_folder('Logs/ml_results/TRAIN_LABEL')
    MachineLearning.set_image_validate_folder('Logs/ml_results/VAL_INPUT')
    MachineLearning.set_label_validate_folder('Logs/ml_results/VAL_LABEL')
    MachineLearning.clear_model_trained()
    MachineLearning.do_semseg_base(model="vgg_unet", input_height=height, input_width=width, n_classes=2, epochs=15,
                                   verify_dataset=False, steps_per_epoch=60, val_steps_per_epoch=18, optimizer_name='adam', batch_size=8)

    # Application.set_input_image_folder('TestData/TMBuD/img/TEST/png')
    # Application.set_output_image_folder('Logs/application_results_semseg_iou')
    # Application.delete_folder_appl_out()
    # Application.do_get_image_job(port_output_name='RAW')
    # class_names = ["UNKNOWN", "BUILDING", "DOOR", "WINDOW", "SKY", "VEGETATION", "GROUND", "NOISE"]
    #
    # BACKGROUND = (0, 0, 0)
    # SKY = (255, 0, 0)
    # VEGETATION = (0, 255, 0)
    # BUILDING = (125, 125, 0)
    # WINDOW = (0, 255, 255)
    # GROUND = (125, 125, 125)
    # NOISE = (0, 0, 255)
    # DOOR = (0, 125, 125)
    #
    # COLORS = [BACKGROUND, BUILDING, DOOR, WINDOW, SKY, VEGETATION, GROUND, NOISE]
    # Application.do_semseg_base_job(port_input_name='RAW', model='vgg_unet', number_of_classes=8, level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
    #                                save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)
    # Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=8, level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
    #                                save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)
    #
    # Application.create_config_file()
    # Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    # Application.run_application()
    #
    # Benchmarking.run_IoU_benchmark(input_location='Logs/application_results_semseg_iou/', gt_location='TestData/TMBuD/label/TEST/png',
    #                                raw_image='TestData/TMBuD/img/TEST/png',
    #                                jobs_set=['SEMSEG_VGG_UNET_RAW_L0', 'SEMSEG_RESNET50_SEGNET_RAW_L0'],
    #                                class_list_name=class_names, unknown_class=0,
    #                                is_rgb_gt=True, show_only_set_mean_value=True,
    #                                class_list_rgb_value=[0, 87, 110, 225, 29, 149, 125, 76])

    Utils.close_files()

def prepare_psb_data(set, w_org, h_org):
    Application.set_output_image_folder('Logs/application_input')
    Application.set_input_image_folder(set)
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_image_crop_job(port_input_name='RAW', is_rgb=True, with_resize=True, new_width=1012, new_height=252,
                                  start_width_percentage=0, end_width_percentage=100, start_height_percentage=21, end_height_percentage=60)
    Application.do_resize_image_job(port_input_name='CROPPED_RAW', new_height=h_org, new_width=w_org, level=CONFIG.PYRAMID_LEVEL.LEVEL_LC0)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()


def main():
    Application.set_output_image_folder('Logs/application')
    # Application.set_input_image_folder('Logs/application_input/RESIZED_1280x320_CROPPED_RAW_LC1')
    Application.set_input_image_folder('TestData/psb/intersect')
    Application.delete_folder_appl_out()

    class_names = ["NON-ROAD", "ROAD"]
    COLORS = [(0, 0, 255), (255, 0, 255)]

    # old approach
    # edge = Application.do_first_order_derivative_operators(port_input_name=grey, operator=CONFIG.FILTERS.PREWITT_5x5)
    # filtered = Application.do_matrix_intersect_job(port_input_name=edge, port_input_mask=croped_filtered)
    # thr_edge = Application.do_image_threshold_job(port_input_name=filtered, input_value=50, input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY)
    # thin_edge = Application.do_thinning_guo_hall_image_job(port_input_name=thr_edge)
    # labeled = Application.do_edge_label_job(port_input_name=thin_edge)

    Application.do_get_image_job(port_output_name='RAW')
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')

    # semseg = Application.do_semseg_base_job(port_input_name='RAW', model='vgg_unet', number_of_classes=2,
    #                                save_img_augmentation=True, save_overlay=True, save_legend_in_image=True,
    #                                list_class_name=class_names, list_colors_to_use=COLORS)
    # dilated_semseg = Application.do_image_morphological_dilation_job(port_input_name=semseg, kernel_size=7, input_iteration=2)
    # croped_filtered = Application.do_image_crop_job(port_input_name=dilated_semseg, is_rgb=False,
    #                                                 start_width_percentage=0, end_width_percentage=100, start_height_percentage=50, end_height_percentage=100)
    # filtered_grey = Application.do_gaussian_blur_image_job(port_input_name=grey, sigma=2)
    # filtered = Application.do_matrix_intersect_job(port_input_name=filtered_grey, port_input_mask=croped_filtered)
    # Application.do_ed_lines_mod_job(port_input_name=filtered, min_line_length=20, gradient_thr=10, anchor_thr=5,
    #                                 line_fit_err_thr=1,
    #                                 operator=CONFIG.FILTERS.ORHEI_DILATED_5x5,
    #                                 max_edges=5000, max_points_edge=1000,
    #                                 max_lines=5000, max_points_line=1000,
    #                                 port_edges_name_output='EDGES', port_edge_map_name_output='EDGE_IMG',
    #                                 port_lines_name_output='LINES', port_lines_img_output='LINES_IMG')

    # comment this to switch back
    Application.do_ed_lines_mod_job(port_input_name=grey, min_line_length=20, gradient_thr=10, anchor_thr=5,
                                    line_fit_err_thr=1,
                                    operator=CONFIG.FILTERS.ORHEI_DILATED_5x5,
                                    max_edges=5000, max_points_edge=1000,
                                    max_lines=5000, max_points_line=1000,
                                    port_edges_name_output='EDGES', port_edge_map_name_output='EDGE_IMG',
                                    port_lines_name_output='LINES', port_lines_img_output='LINES_IMG')

    horizontal_line, horizontal_line_img = Application.do_line_theta_filtering_job(port_input_name='LINES', theta_value=0, deviation_theta=0.005, nr_lines=5000, nr_pt_line=1000)

    sb_lines, sb_img = Application.do_sb_detection_from_lines_job(port_input_name=horizontal_line,
                                                                  min_gap_horizontal_lines=1, max_gap_horizontal_lines=50,
                                                                  min_gap_vertical_lines=1, max_gap_vertical_lines=50)

    final = Application.do_blending_images_job(port_input_name_1='RAW', port_input_name_2=horizontal_line_img, alpha=0.7)
    final_2 = Application.do_blending_images_job(port_input_name_1='RAW', port_input_name_2=sb_img, alpha=0.7)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    # Application.configure_show_pictures(ports_to_show=[final + '_L0', final_2 + '_L0'], time_to_show=10)
    Application.configure_show_pictures(ports_to_show=[final_2 + '_L0'], time_to_show=10)
    Application.run_application()

    Utils.close_files()


if __name__ == "__main__":
    w = 640
    h = 160
    set = 'TestData/psb/set_fina'

    w_org = w * 2
    h_org = h * 2

    # prepare_dataset()
    # Utils.reopen_files()
    # main_training_data(width=w, height=h)
    # Utils.reopen_files()
    # main_training_label(width=w, height=h)
    # Utils.reopen_files()
    # train_model(width=w, height=h)
    # Utils.reopen_files()
    # prepare_psb_data(set=set, w_org=w_org, h_org=h_org)
    # Utils.reopen_files()
    main()
