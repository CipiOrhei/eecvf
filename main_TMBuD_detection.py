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

def prepare_dataset_labels(dataset_in, dataset_out, LabelMe_COLORS, LabelMe_BDT_CORRELATION, BDT_COLORS, BDT_CLASSES):
    """
    Please download https://github.com/cvjena/labelmefacade for this experiment
    This function will correlate the images from LabelMe Facade to TMBuD dataset
    :param height: height of image to train
    :param width: width of image to train
    :return: None
    """
    Application.set_input_image_folder(dataset_in)
    Application.set_output_image_folder(dataset_out)
    Application.do_get_image_job(port_output_name='RAW')
    #                VARIOUS      BUILDING        CAR            DOOR          PAVEMENT         ROAD           SKY       VEGETATION      WINDOW
    # LabelMe_COLORS = [(0, 0, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (0, 64, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0)]
    # LabelMe_BDT_CORRELATION = [0, 1, 2, 1, 3, 3, 4, 2, 1]
    #            [BACKGROUND,  BUILDING,       SKY,         GROUND,         NOISE]
    # BDT_COLORS = [(0, 0, 0), (125, 125, 0), (255, 0, 0), (125, 125, 125), (0, 0, 255)]
    # BDT_CLASSES = [0,           1,              2,              3,              4]
    Application.do_class_correlation(port_input_name='RAW', port_output_name='BDT_LABELS', class_list_in=LabelMe_COLORS, class_list_out=LabelMe_BDT_CORRELATION)
    Application.do_class_correlation(port_input_name='BDT_LABELS', port_output_name='BDT_LABELS_PNG', class_list_in=BDT_CLASSES, class_list_out=BDT_COLORS)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()

def prepare_dataset_img(dataset_out, dataset_img_input):
    Application.set_input_image_folder(dataset_img_input)
    Application.set_output_image_folder(dataset_out)
    Application.do_get_image_job(port_output_name='RAW')
    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()

def prepare_dataset_TMBuD(COLORS_TMBuD, TMBuD_CORRELATION):
    """
    Please download https://github.com/cvjena/labelmefacade for this experiment
    This function will correlate the images from LabelMe Facade to TMBuD dataset
    :param height: height of image to train
    :param width: width of image to train
    :return: None
    """

    dataset_input_labels_tmbud = r'c:\repos\eecvf_git\TestData\TMBuD\parsed_dataset\label_full\classes'
    dataset_processed_tmbud = 'Logs/TMBuD/labels'

    Application.set_input_image_folder(dataset_input_labels_tmbud)
    Application.set_output_image_folder(dataset_processed_tmbud)
    Application.do_get_image_job(port_output_name='RAW', direct_grey=False)
    #                VARIOUS      BUILDING        CAR            DOOR          PAVEMENT         ROAD           SKY       VEGETATION      WINDOW
    # LabelMe_COLORS = [(0, 0, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (0, 64, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0)]
    # LabelMe_BDT_CORRELATION = [0, 1, 2, 1, 3, 3, 4, 2, 1]
    #            [BACKGROUND,  BUILDING,       SKY,         GROUND,         NOISE]
    # BDT_COLORS = [(0, 0, 0), (125, 125, 0), (255, 0, 0), (125, 125, 125), (0, 0, 255)]
    # BDT_CLASSES = [0,           1,              2,              3,              4]

    Application.do_class_correlation(port_input_name='RAW', port_output_name='BDT_LABELS', class_list_in=[0, 1, 2, 3, 4, 5, 6, 7], class_list_out=TMBuD_CORRELATION)
    Application.do_class_correlation(port_input_name='BDT_LABELS', port_output_name='BDT_LABELS_PNG', class_list_in=BDT_CLASSES, class_list_out=BDT_COLORS)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()

def main_training_data(height, width, data_input_img):
    Application.set_output_image_folder('Logs/application_results_ml_raw')
    Application.set_input_image_folder(data_input_img)
    # Application.delete_folder_appl_out()
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
    # Application.delete_folder_appl_out()
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


def main_training_label(height, width, dataset_input):
    Application.set_output_image_folder('Logs/application_results_ml_labels')
    Application.set_input_image_folder(dataset_input)
    # Application.delete_folder_appl_out()

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
    # Application.delete_folder_appl_out()
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


def train_model(height, width, n_classes, epochs, steps_per_epoch, val_steps_per_epoch, batch_size, class_names, COLORS, validate_input, validate_gt, class_list_rgb_value):
    MachineLearning.set_image_input_folder('Logs/ml_results/TRAIN_INPUT')
    MachineLearning.set_label_input_folder('Logs/ml_results/TRAIN_LABEL')
    MachineLearning.set_image_validate_folder('Logs/ml_results/VAL_INPUT')
    MachineLearning.set_label_validate_folder('Logs/ml_results/VAL_LABEL')
    MachineLearning.clear_model_trained()
    # MachineLearning.do_semseg_base(model="vgg_unet", input_height=height, input_width=width, n_classes=n_classes, epochs=epochs,
    #                                verify_dataset=False, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch, optimizer_name='adam', batch_size=batch_size)
    MachineLearning.do_semseg_base(model="resnet50_segnet", input_height=height, input_width=width, n_classes=n_classes, epochs=epochs,
                                   verify_dataset=False, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch, optimizer_name='adam', batch_size=batch_size)
    Application.set_input_image_folder(validate_input)
    Application.set_output_image_folder('Logs/application_results_semseg_iou')
    # Application.delete_folder_appl_out()
    Application.do_get_image_job(port_output_name='RAW')

    # class_names = ["UNKNOWN", "BUILDING", "SKY", "GROUND", "NOISE"]

    # BACKGROUND = (0, 0, 0)
    # SKY = (255, 0, 0)
    # VEGETATION = (0, 255, 0)
    # BUILDING = (125, 125, 0)
    # WINDOW = (0, 255, 255)
    # GROUND = (125, 125, 125)
    # NOISE = (0, 0, 255)
    # DOOR = (0, 125, 125)
    #
    # COLORS = [BACKGROUND, BUILDING, SKY, GROUND, NOISE]

    # Application.do_semseg_base_job(port_input_name='RAW', model='vgg_unet', number_of_classes=n_classes, level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
    #                                save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)
    Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=n_classes, level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                   save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Benchmarking.run_IoU_benchmark(input_location='Logs/application_results_semseg_iou/',
                                   gt_location=validate_gt,
                                   raw_image=validate_input,
                                   # jobs_set=['SEMSEG_VGG_UNET_RAW_L0', 'SEMSEG_RESNET50_SEGNET_RAW_L0'],
                                   jobs_set=['SEMSEG_RESNET50_SEGNET_RAW_L0'],
                                   # jobs_set=['SEMSEG_VGG_UNET_RAW_L0'],
                                   class_list_name=class_names, unknown_class=0,
                                   is_rgb_gt=True, show_only_set_mean_value=True,
                                   class_list_rgb_value=class_list_rgb_value)
                                   # class_list_rgb_value=[0, 87])

    Utils.close_files()


def main_bow_create(building_classes, desc_list, diff_list, desc_size_list, nOctaves_list, nLayes_list, thr_list, thr_akaze_list, dictionarySize_list, class_in, class_out, class_names, COLORS):
    """
    Main function of framework Please look in example_main for all functions you can use
    """

    # class_names = ["UNKNOWN", "BUILDING", "SKY", "GROUND", "NOISE"]
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
    # COLORS = [BACKGROUND, BUILDING, SKY, GROUND, NOISE]


    Application.set_input_image_folder('TestData/TMBuD/parsed_dataset/v3_2/TRAIN')
    Application.set_output_image_folder('Logs/application_results')
    Application.delete_folder_appl_out()

    # grey = Application.do_get_image_job(port_output_name='GRAY_RAW', direct_grey=True)
    Application.do_get_image_job(port_output_name='RAW', direct_grey=False)
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')
    # filtered = Application.do_guided_filter_job(port_input_name=grey, radius=2, regularization=0.4, is_rgb=False)
    # filtered = Application.do_median_blur_job(port_input_name=grey, kernel_size=5, is_rgb=False)

    list_to_eval = list()

    for desc in desc_list:
        for diff in diff_list:
            for desc_size in desc_size_list:
                for nOctaves in nOctaves_list:
                    for nLayes in nLayes_list:
                        for thr in thr_list:
                            for thr_akaze in thr_akaze_list:
                                for dict_size in dictionarySize_list:
                                    semseg_image = Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=len(class_names), level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                                                                  save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)

                                    binary_mask = Application.do_class_correlation(port_input_name=semseg_image, class_list_in=class_in, class_list_out=class_out)

                                    kp, des, img = Application.do_a_kaze_job(port_input_name=grey, descriptor_channels=1, mask_port_name=binary_mask,
                                                                             descriptor_size=desc_size, descriptor_type=desc, diffusivity=diff,
                                                                             threshold=thr_akaze, nr_octaves=nOctaves, nr_octave_layers=nLayes)

                                    bow = Application.do_tmbud_bow_job(port_to_add=des, dictionary_size=dict_size, number_classes=building_classes)

    Application.create_config_file()
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save='ALL')
    # Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save=[])
    Application.run_application()

    Utils.close_files()


def main_bow_inquiry(building_classes, desc_list, diff_list, desc_size_list, nOctaves_list, nLayes_list, thr_list, thr_akaze_list, dictionarySize_list, class_in, class_out, class_names, COLORS):
    """
    Main function of framework Please look in example_main for all functions you can use
    """

    # class_names = ["UNKNOWN", "BUILDING", "SKY", "GROUND", "NOISE"]
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
    # COLORS = [BACKGROUND, BUILDING, SKY, GROUND, NOISE]

    list_to_eval = list()

    Application.set_input_image_folder('TestData/TMBuD/parsed_dataset/v3_2/TEST')
    Application.set_output_image_folder('Logs/query_application')
    # Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW', direct_grey=False)
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')
    # filtered = Application.do_guided_filter_job(port_input_name=grey, radius=2, regularization=0.4, is_rgb=False)
    # filtered = Application.do_median_blur_job(port_input_name=grey, kernel_size=5, is_rgb=False)


    for desc in desc_list:
        for diff in diff_list:
            for desc_size in desc_size_list:
                for nOctaves in nOctaves_list:
                    for nLayes in nLayes_list:
                        for thr in thr_list:
                            for thr_akaze in thr_akaze_list:
                                for dict_size in dictionarySize_list:
                                    semseg_image = Application.do_semseg_base_job(port_input_name='RAW', model='resnet50_segnet', number_of_classes=3, level=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                                                                  save_img_augmentation=True, save_overlay=True, save_legend_in_image=True, list_class_name=class_names, list_colors_to_use=COLORS)

                                    binary_mask = Application.do_class_correlation(port_input_name=semseg_image, class_list_in=class_in, class_list_out=class_out)

                                    kp, des, img = Application.do_a_kaze_job(port_input_name=grey, descriptor_channels=1, mask_port_name=binary_mask,
                                                                             descriptor_size=desc_size, descriptor_type=desc, diffusivity=diff,
                                                                             threshold=thr_akaze, nr_octaves=nOctaves, nr_octave_layers=nLayes)

                                    final = Application.do_tmbud_bow_inquiry_flann_job(port_to_inquiry=des, flann_thr=thr, saved_to_npy=True, number_classes=building_classes,
                                                                                       location_of_bow='Logs/application_results',
                                                                                       bow_port='ZuBuD_BOW_' + dict_size.__str__() + '_' + des + '_L0')

                                    list_to_eval.append(final + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save='ALL')
    # Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save=[])
    # Application.run_application()

    Benchmarking.run_CBIR_ZuBuD_benchmark(input_location='Logs/query_application/',
                                          gt_location='TestData/TMBuD/parsed_dataset/v3_2/TMBuD_groundtruth.txt',
                                          raw_image='TestData/TMBuD/parsed_dataset/v3_2/TEST',
                                          jobs_set=list_to_eval)

    Utils.close_files()


if __name__ == "__main__":
    w = 320
    h = 512

    #            [BACKGROUND,  BUILDING,      NOISE]
    BDT_COLORS = [(0, 0, 0), (125, 125, 0), (0, 0, 255)]
    BDT_CLASSES = [0,           1,              2]

    dataset_processed = 'Logs/bulk_data'
    dataset_input_img = 'Logs/bulk_data/img/RAW_L0'
    dataset_input_labels_processed = 'Logs/bulk_data/labels/BDT_LABELS_L0'

    # dataset_input_labels = r'c:\repos\eecvf_git\TestData\building_labels_database\eTRIMS\annotations\08_etrims-ds'
    # dataset_input_img = r'c:\repos\eecvf\TestData\building_labels_database\eTRIMS\images\image'
    # #                          VARIOUS      BUILDING        CAR            DOOR          PAVEMENT         ROAD           SKY       VEGETATION      WINDOW
    # LabelMe_COLORS =          [(0, 0, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (0, 64, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0)]
    # LabelMe_BDT_CORRELATION = [     0,          1,           2,             1,           2,                 2,           2,              2,          1]
    # prepare_dataset_img(dataset_img_input=dataset_input_img, dataset_out=dataset_processed + '/img')
    # Utils.reopen_files()
    # prepare_dataset_labels(dataset_in=dataset_input_labels, dataset_out=dataset_processed + '/labels', LabelMe_COLORS=LabelMe_COLORS, LabelMe_BDT_CORRELATION=LabelMe_BDT_CORRELATION, BDT_COLORS=BDT_COLORS, BDT_CLASSES=BDT_CLASSES)
    # Utils.reopen_files()
    #
    # dataset_input_labels = r'c:\repos\eecvf_git\TestData\building_labels_database\LabelMeFacade\labels'
    # dataset_input_img = r'c:\repos\eecvf\TestData\building_labels_database\LabelMeFacade\images'
    # #                          VARIOUS      BUILDING        CAR            DOOR          PAVEMENT         ROAD           SKY       VEGETATION      WINDOW
    # eTRIMS_COLORS =          [(0, 0, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (0, 64, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0)]
    # eTRIMS_BDT_CORRELATION = [     0,          1,           2,             1,           2,                 2,           2,              2,          1]
    # prepare_dataset_img(dataset_img_input=dataset_input_img, dataset_out=dataset_processed + '/img')
    # Utils.reopen_files()
    # prepare_dataset_labels(dataset_in=dataset_input_labels, dataset_out=dataset_processed + '/labels', LabelMe_COLORS=eTRIMS_COLORS, LabelMe_BDT_CORRELATION=eTRIMS_BDT_CORRELATION, BDT_COLORS=BDT_COLORS, BDT_CLASSES=BDT_CLASSES )
    # Utils.reopen_files()
    #
    # dataset_input_labels = r'c:\repos\eecvf_git\TestData\TMBuD\parsed_dataset\label_full\png'
    # dataset_input_img = r'c:\repos\eecvf_git\TestData\TMBuD\parsed_dataset\img_label_full\png'
    # # class_names = [           "UNKNOWN", "BUILDING",      "DOOR",         "WINDOW",      "SKY",   "VEGETATION",   "GROUND",       "NOISE"]
    # TMBuD_COLORS =       [(     0, 0, 0), (125, 125, 0), (0, 125, 125), (0, 255, 255), (255, 0, 0), (0, 255, 0), (125, 125, 125), (0, 0, 255)]
    # TMBuD_BDT_CORRELATION = [     0,          1,                1,             1,           2,           2,           2,              2]
    # prepare_dataset_img(dataset_img_input=dataset_input_img, dataset_out=dataset_processed + '/img')
    # Utils.reopen_files()
    # prepare_dataset_labels(dataset_in=dataset_input_labels, dataset_out=dataset_processed + '/labels', LabelMe_COLORS=TMBuD_COLORS, LabelMe_BDT_CORRELATION=TMBuD_BDT_CORRELATION, BDT_COLORS=BDT_COLORS, BDT_CLASSES=BDT_CLASSES)
    # Utils.reopen_files()

    # main_training_data(width=w, height=h, data_input_img=dataset_input_img)
    # Utils.reopen_files()
    # main_training_label(width=w, height=h, dataset_input=dataset_input_labels_processed)
    # Utils.reopen_files()

    # class_names = ["UNKNOWN", "BUILDING", "DOOR", "WINDOW", "SKY", "VEGETATION", "GROUND", "NOISE"]
    # COLORS_TMBuD = [(     0, 0, 0), (125, 125, 0), (0, 125, 125), (0, 255, 255), (255, 0, 0), (0, 255, 0), (125, 125, 125), (0, 0, 255)]
    # TMBuD_CORRELATION = [     0,          1,           1,             1,           2,                 2,           2,              2]
    # prepare_dataset_TMBuD(COLORS_TMBuD=COLORS_TMBuD, TMBuD_CORRELATION=TMBuD_CORRELATION)
    # Utils.reopen_files()

    n_classes = 3
    epochs = 50
    batch_size = 4
    train_nr_images = 3673
    val_nr_images = 136
    steps_per_epoch = int((train_nr_images/epochs)/batch_size)
    val_steps_per_epoch = int(val_nr_images/batch_size)
    class_names = ["UNKNOWN", "BUILDING", "NOISE"]
    COLORS = [(0, 0, 0), (125, 125, 0), (0, 0, 255)]
    validate_input = 'TestData/TMBuD/parsed_dataset/img_label_full/png'
    validate_gt = 'Logs/TMBuD/labels/BDT_LABELS_PNG_L0'
    class_list_rgb_value = [0, 87, 76]
    # train_model(width=w, height=h, n_classes=n_classes, epochs=epochs, val_steps_per_epoch=val_steps_per_epoch, batch_size=batch_size, steps_per_epoch=steps_per_epoch, class_names=class_names, COLORS=COLORS,
    #             validate_input=validate_input, validate_gt=validate_gt, class_list_rgb_value=class_list_rgb_value, )
    # Utils.reopen_files()

    # desc_list = [cv2.AKAZE_DESCRIPTOR_KAZE, cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT, cv2.AKAZE_DESCRIPTOR_MLDB, cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT]
    desc_list = [cv2.AKAZE_DESCRIPTOR_KAZE]
    # diff_list = [cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2, cv2.KAZE_DIFF_CHARBONNIER, cv2.KAZE_DIFF_WEICKERT]
    diff_list = [cv2.KAZE_DIFF_PM_G1]
    # desc_size_list = [0, 8, 16, 32, 64, 128]
    desc_size_list = [64]
    nOctaves_list = [4]
    nLayes_list = [5]
    thr_list = [0.82]
    # thr_akaze_list = [0.0010, 0.0011, 0.0012, 0.0013]
    thr_akaze_list = [0.0012]
    # dictionarySize_list = [375, 400, 425]
    dictionarySize_list = [400]
    class_in = [0, 1, 2]
    class_out = [0, 1, 0]

    # main_bow_create(building_classes=105, desc_list=desc_list, diff_list=diff_list, desc_size_list=desc_size_list,
    #                 nOctaves_list=nOctaves_list, nLayes_list=nLayes_list, thr_list=thr_list, thr_akaze_list=thr_akaze_list, dictionarySize_list=dictionarySize_list,
    #                 class_in=class_in, class_out=class_out, class_names=class_names, COLORS=COLORS)
    # Utils.reopen_files()
    main_bow_inquiry(building_classes=105, desc_list=desc_list, diff_list=diff_list, desc_size_list=desc_size_list,
                    nOctaves_list=nOctaves_list, nLayes_list=nLayes_list, thr_list=thr_list, thr_akaze_list=thr_akaze_list, dictionarySize_list=dictionarySize_list,
                    class_in=class_in, class_out=class_out, class_names=class_names, COLORS=COLORS)
