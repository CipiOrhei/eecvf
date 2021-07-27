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

"""
This module is an example to use the EECVF with video input stream.
"""


def prepare_LabelMe_dataset(height, width):
    """
    Please download https://github.com/cvjena/labelmefacade for this experiment
    This function will correlate the images from LabelMe Facade to TMBuD dataset
    :param height: height of image to train
    :param width: width of image to train
    :return: None
    """
    Application.set_input_image_folder('TestData/building_labels_database/LabelMeFacade/labels')
    Application.set_output_image_folder('Logs/LabelMeFacade/labels')
    Application.do_get_image_job(port_output_name='RAW')
    #                VARIOUS      BUILDING        CAR            DOOR          PAVEMENT         ROAD           SKY       VEGETATION      WINDOW
    LabelMe_COLORS = [(0, 0, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (0, 64, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0)]
    LabelMe_BDT_CORRELATION = [0, 1, 7, 2, 6, 6, 4, 5, 3]
    #            [BACKGROUND,  BUILDING,       DOOR,           WINDOW,         SKY,    VEGETATION,     GROUND,         NOISE]
    BDT_COLORS = [(0, 0, 0), (125, 125, 0), (0, 125, 125), (0, 255, 255), (255, 0, 0), (0, 255, 0), (125, 125, 125), (0, 0, 255)]
    BDT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]
    Application.do_class_correlation(port_input_name='RAW', port_output_name='BDT_LABELS', class_list_in=LabelMe_COLORS, class_list_out=LabelMe_BDT_CORRELATION)
    Application.do_class_correlation(port_input_name='BDT_LABELS', port_output_name='BDT_LABELS_PNG', class_list_in=BDT_CLASSES, class_list_out=BDT_COLORS)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()

def main_training_data(height, width):
    Application.set_output_image_folder('Logs/application_results_ml_raw')
    Application.set_input_image_folder(r'c:\repos\eecvf\TestData\building_labels_database\LabelMeFacade\images')
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
    Application.set_input_image_folder('Logs/LabelMeFacade/labels/BDT_LABELS_L0')
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
    MachineLearning.do_semseg_base(model="vgg_unet", input_height=height, input_width=width, n_classes=8, epochs=70,
                                   verify_dataset=False, steps_per_epoch=20, val_steps_per_epoch=58, optimizer_name='adam', batch_size=8)
    MachineLearning.do_semseg_base(model="resnet50_segnet", input_height=height, input_width=width, n_classes=8, epochs=70,
                                   verify_dataset=False, steps_per_epoch=58, val_steps_per_epoch=117, optimizer_name='adam', batch_size=4)
    Application.set_input_image_folder('TestData/TMBuD/img/TEST/png')
    Application.set_output_image_folder('Logs/application_results_semseg_iou')
    Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')
    class_names = ["UNKNOWN", "BUILDING", "DOOR", "WINDOW", "SKY", "VEGETATION", "GROUND", "NOISE"]

    BACKGROUND = (0, 0, 0)
    SKY = (255, 0, 0)
    VEGETATION = (0, 255, 0)
    BUILDING = (125, 125, 0)
    WINDOW = (0, 255, 255)
    GROUND = (125, 125, 125)
    NOISE = (0, 0, 255)
    DOOR = (0, 125, 125)

    COLORS = [BACKGROUND, BUILDING, DOOR, WINDOW, SKY, VEGETATION, GROUND, NOISE]
    Application.do_semseg_base_job(port_input_name='RAW', model='vgg_unet', number_of_classes=8, level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                   save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)
    Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=8, level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                   save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Benchmarking.run_IoU_benchmark(input_location='Logs/application_results_semseg_iou/', gt_location='TestData/TMBuD/label/TEST/png',
                                   raw_image='TestData/TMBuD/img/TEST/png',
                                   jobs_set=['SEMSEG_VGG_UNET_RAW_L0', 'SEMSEG_RESNET50_SEGNET_RAW_L0'],
                                   class_list_name=class_names, unknown_class=0,
                                   is_rgb_gt=True, show_only_set_mean_value=True,
                                   class_list_rgb_value=[0, 87, 110, 225, 29, 149, 125, 76])

    Utils.close_files()


def main_process_edges():
    Application.set_output_image_folder('Logs/application_results')
    Application.set_input_image_folder('TestData/TMBuD/img/TEST/png')
    Application.delete_folder_appl_out()

    class_names = ["UNKNOWN", "BUILDING", "DOOR", "WINDOW", "SKY", "VEGETATION", "GROUND", "NOISE"]
    classes = [0, 1, 2, 3, 4, 5, 6, 7]
    class_correlation = [0, 1, 1, 1, 0, 0, 0, 0]

    BACKGROUND = (0, 0, 0)
    SKY = (255, 0, 0)
    VEGETATION = (0, 255, 0)
    BUILDING = (125, 125, 0)
    WINDOW = (0, 255, 255)
    GROUND = (125, 125, 125)
    NOISE = (0, 0, 255)
    DOOR = (0, 125, 125)

    COLORS = [BACKGROUND, BUILDING, DOOR, WINDOW, SKY, VEGETATION, GROUND, NOISE]

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_pyramid_level_down_job(port_input_name='RAW', number_of_lvl=1, is_rgb=True)
    levels = [CONFIG.PYRAMID_LEVEL.LEVEL_0, CONFIG.PYRAMID_LEVEL.LEVEL_1]

    edge_list = list()

    for level in levels:
        smoothing = list()
        semseg_output = list()
        semseg_output.append(Application.do_semseg_base_job(port_input_name='RAW', model='vgg_unet', number_of_classes=8, level=level,
                                                            save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS))
        semseg_output.append(Application.do_semseg_base_job(port_input_name="RAW", model='resnet50_segnet', number_of_classes=8, level=level,
                                                            save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS))
        for k in [4, 8]:
            for s in [90, 130]:
                smoothing.append(Application.do_bilateral_filter_job(port_input_name='RAW', distance=k, sigma_colors=s, sigma_space=s, port_output_name='BI_K={k}_S={s}'.format(k=k, s=s), is_rgb=True, level=level))
            for kappa in [0.2, 0.6]:
                for niter in [2, 6]:
                    smoothing.append(Application.do_anisotropic_diffusion_filter_job(port_input_name='RAW', alpha=alpha, kappa=kappa, niter=niter, port_output_name='AD_A={a}_K={k}_N={n}'.format(a=alpha, k=kappa, n=niter).replace('.', '_'), is_rgb=True, level=level))
        for img in smoothing:
            for semseg_img in semseg_output:
                semseg_correlation = Application.do_class_correlation(port_input_name=semseg_img, class_list_in=classes, class_list_out=class_correlation, level=level)
                grey = Application.do_grayscale_transform_job(port_input_name=img, level=level)
                filtered = Application.do_matrix_intersect_job(port_input_name=grey, port_input_mask=semseg_correlation, level=level)
                Application.do_median_pixel_image_job(port_input_name=filtered)
                edge_1 = Application.do_canny_otsu_median_sigma_job(port_input_name=filtered, edge_detector=CONFIG.FILTERS.ORHEI_DILATED_7x7, do_blur=False, level=level, is_rgb=False)
                edge_2 = Application.do_shen_castan_job(port_input_name=filtered, laplacian_threhold=1, smoothing_factor=0.9, ratio=0.1, thinning_factor=0.5, zc_window_size=5, level=level)
                edge_3, edge_4 = Application.do_edge_drawing_mod_job(port_input_name=filtered, gradient_thr=30, anchor_thr=10, scan_interval=1, operator=CONFIG.FILTERS.ORHEI_DILATED_7x7, level=level)
                if level == CONFIG.PYRAMID_LEVEL.LEVEL_0:
                    edge_list.extend([edge_1, edge_2, edge_3])
                elif level == CONFIG.PYRAMID_LEVEL.LEVEL_1:
                    for edge in [edge_1, edge_2, edge_3]:
                        Application.do_pyramid_level_up_job(port_input_name=edge, port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1, number_of_lvl=1, port_output_name='EXP_' + edge)
                        thr_restore_edge = Application.do_image_threshold_job(port_input_name='EXP_' + edge, input_value=1, input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY, port_output_name='THR_EXP_' + edge, level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
                        edge_list.append(Application.do_thinning_guo_hall_image_job(port_input_name=thr_restore_edge, port_output_name='REC_' + edge, level=CONFIG.PYRAMID_LEVEL.LEVEL_0))

    Application.create_config_file()

    for port in range(len(edge_list.copy())):
        edge_list[port] += '_L0'

    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    Application.run_application()
    Benchmarking.run_FOM_benchmark(input_location='Logs/application_results/', gt_location='TestData/TMBuD/edge/TEST/png', raw_image='TestData/TMBuD/img/TEST/png', jobs_set=edge_list)
    Utils.create_latex_fom_table(number_decimal=3, number_of_series=25)
    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results', gt_location='TestData/TMBuD/edge/TEST/mat', raw_image='TestData/TMBuD/img/TEST/png',jobs_set=edge_list, do_thinning=False)
    Utils.plot_first_cpm_results( level='L0', order_by='f1', name='ROC_edge_results',
                                 list_of_data=edge_list, number_of_series=25, self_contained_list=True,
                                 save_plot=True, show_plot=False)
    Utils.plot_avg_time_jobs(save_plot=True, show_legend=False)

    Utils.close_files()


if __name__ == "__main__":
    w = 320
    h = 512
    prepare_LabelMe_dataset(width=w, height=h)
    Utils.reopen_files()
    main_training_data(width=w, height=h)
    Utils.reopen_files()
    main_training_label(width=w, height=h)
    Utils.reopen_files()
    train_model(width=w, height=h)
    Utils.reopen_files()
    main_process_edges()