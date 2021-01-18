from config_main import CUDA_GPU
########################################################################################################################
# service jobs
########################################################################################################################
from Application.Config.create_config import create_config_file
from Application.Config.service_job_create import set_input_image_folder
from Application.Config.service_job_create import set_output_image_folder
from Application.Config.service_job_create import set_input_video
from Application.Config.service_job_create import set_number_waves
from Application.Config.service_job_create import set_input_camera_video
from Application.Config.service_job_create import configure_save_pictures
from Application.Config.service_job_create import configure_show_pictures
from Application.Config.service_job_create import delete_folder_appl_out
from Application.Config.service_job_create import create_list_ports_with_word
from Application.Config.service_job_create import create_list_ports_start_with_word
from Application.Config.service_job_create import create_folder_from_list_ports
from .run_appl import run_application
########################################################################################################################
# Input jobs
########################################################################################################################
from Application.Config.job_create import do_get_image_job
from Application.Config.job_create import do_get_satellite_image_job
from Application.Config.job_create import do_get_video_job
from Application.Config.job_create import do_get_video_capture_job
########################################################################################################################
# Pyramid level processing jobs
########################################################################################################################
from Application.Config.job_create import do_pyramid_level_down_job
from Application.Config.job_create import do_pyramid_level_up_job
########################################################################################################################
# Image processing jobs
########################################################################################################################
from Application.Config.job_create import do_max_pixel_image_job
from Application.Config.job_create import do_median_pixel_image_job
from Application.Config.job_create import do_mean_pixel_image_job
from Application.Config.job_create import do_add_gaussian_blur_noise_job
from Application.Config.job_create import do_add_salt_pepper_noise
from Application.Config.job_create import do_add_speckle_noise
from Application.Config.job_create import do_grayscale_transform_job
from Application.Config.job_create import do_image_complement_job
from Application.Config.job_create import do_number_edge_pixels
from Application.Config.job_create import do_image_crop_job
from Application.Config.job_create import do_rotate_image_job
from Application.Config.job_create import do_flip_image_job
from Application.Config.job_create import do_zoom_image_job
from Application.Config.job_create import do_contrast_brightness_change_image_job
from Application.Config.job_create import do_gamma_correction_image_job
from Application.Config.job_create import do_pixelate_image_job
########################################################################################################################
# Image blurring jobs
########################################################################################################################
from Application.Config.job_create import do_gaussian_blur_image_job
from Application.Config.job_create import do_median_blur_job
from Application.Config.job_create import do_mean_blur_job
from Application.Config.job_create import do_conservative_filter_job
from Application.Config.job_create import do_bilateral_filter_job
from Application.Config.job_create import do_guided_filter_job
from Application.Config.job_create import do_l0_gradient_minimization_filter_job
from Application.Config.job_create import do_anisotropic_diffusion_filter_job
from Application.Config.job_create import do_crimmins_job
from Application.Config.job_create import do_sharpen_filter_job
from Application.Config.job_create import do_unsharp_filter_job
from Application.Config.job_create import do_isef_filter_job
from Application.Config.job_create import do_motion_blur_filter_job
########################################################################################################################
# Image morphology jobs
########################################################################################################################
from Application.Config.job_create import do_image_morphological_erosion_job
from Application.Config.job_create import do_image_morphological_dilation_job
from Application.Config.job_create import do_image_morphological_open_job
from Application.Config.job_create import do_image_morphological_close_job
from Application.Config.job_create import do_image_morphological_edge_gradient_job
from Application.Config.job_create import do_image_morphological_top_hat_job
from Application.Config.job_create import do_image_morphological_black_hat_job
from Application.Config.job_create import do_morphological_hit_and_miss_transformation_job
from Application.Config.job_create import do_image_morphological_cv2_job
from Application.Config.job_create import do_morphological_thinning_job
########################################################################################################################
# Kernel processing jobs
########################################################################################################################
from Application.Config.job_create import do_kernel_convolution_job
from Application.Config.job_create import do_deriche_kernel_convolution_job
from Application.Config.job_create import do_kernel_cross_convolution_job
from Application.Config.job_create import do_kernel_frei_chen_convolution_job
from Application.Config.job_create import do_kernel_navatia_babu_convolution_job
########################################################################################################################
# edge detection - magnitude gradient jobs
########################################################################################################################
from Application.Config.job_create import do_gradient_magnitude_job
from Application.Config.job_create import do_first_order_derivative_operators
########################################################################################################################
# edge detection - directional gradient jobs
########################################################################################################################
from Application.Config.job_create import do_gradient_magnitude_cross_job
from Application.Config.job_create import do_gradient_navatia_babu_job
from Application.Config.job_create import do_gradient_frei_chen_job
from Application.Config.job_create import do_frei_chen_edge_job
from Application.Config.job_create import do_navatia_babu_edge_5x5_job
from Application.Config.job_create import do_compass_edge_job
from Application.Config.job_create import do_kirsch_3x3_cross_job
from Application.Config.job_create import do_robinson_3x3_cross_job
from Application.Config.job_create import do_robinson_modified_3x3_cross_job
from Application.Config.job_create import do_prewitt_3x3_cross_job
########################################################################################################################
# edge detection - Canny jobs
########################################################################################################################
from Application.Config.job_create import do_canny_from_kernel_convolution_job
from Application.Config.job_create import do_canny_config_job
from Application.Config.job_create import do_canny_fix_threshold_job
from Application.Config.job_create import do_canny_ratio_threshold_job
from Application.Config.job_create import do_canny_otsu_half_job
from Application.Config.job_create import do_canny_otsu_median_sigma_job
from Application.Config.job_create import do_canny_median_sigma_job
from Application.Config.job_create import do_canny_mean_sigma_job
from Application.Config.job_create import do_deriche_canny_job
########################################################################################################################
# edge detection - second derivative
########################################################################################################################
from Application.Config.job_create import do_laplacian_pyramid_from_img_diff_job
from Application.Config.job_create import do_laplacian_from_img_diff_job
from Application.Config.job_create import do_laplace_job
from Application.Config.job_create import do_log_job
from Application.Config.job_create import do_zero_crossing_job
from Application.Config.job_create import do_zero_crossing_adaptive_window_isef_job
from Application.Config.job_create import do_threshold_hysteresis_isef_job
from Application.Config.job_create import do_shen_castan_job
from Application.Config.job_create import do_marr_hildreth_job
from Application.Config.job_create import do_dog_job
from Application.Config.job_create import do_dob_job
########################################################################################################################
# line/shape detection
########################################################################################################################
from Application.Config.job_create import do_hough_lines_job
from Application.Config.job_create import do_hough_circle_job
########################################################################################################################
# Image threshold jobs
########################################################################################################################
from Application.Config.job_create import do_otsu_job
from Application.Config.job_create import do_image_threshold_job
from Application.Config.job_create import do_image_adaptive_threshold_job
########################################################################################################################
# Skeletonization/thinning jobs
########################################################################################################################
from Application.Config.job_create import do_thinning_guo_hall_image_job
########################################################################################################################
# Line/edge connectivity jobs
########################################################################################################################
from Application.Config.job_create import do_edge_label_job
########################################################################################################################
# Multiple image jobs
########################################################################################################################
from Application.Config.job_create import do_matrix_difference_job
from Application.Config.job_create import do_matrix_difference_1_px_offset_job
from Application.Config.job_create import do_matrix_sum_job
from Application.Config.job_create import do_matrix_bitwise_and_job
from Application.Config.job_create import do_matrix_bitwise_or_job
from Application.Config.job_create import do_matrix_bitwise_or_4_job
from Application.Config.job_create import do_matrix_bitwise_xor_job
########################################################################################################################
# Augmentation jobs
########################################################################################################################
from Application.Config.job_create import do_class_correlation
########################################################################################################################
# U-Net jobs
########################################################################################################################
if CUDA_GPU:
    from Application.Config.job_create import do_u_net_edge
############################################################################################################################################
# Semseg jobs
############################################################################################################################################
if CUDA_GPU:
    from Application.Config.job_create import do_mobilenet_unet_semseg
    from Application.Config.job_create import do_unet_mini_semseg
    from Application.Config.job_create import do_resnet50_unet_semseg
    from Application.Config.job_create import do_u_net_semseg
    from Application.Config.job_create import do_vgg_u_net_semseg
    from Application.Config.job_create import do_semseg_base_job