import numpy as np

import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils


def main_get_img_from_movie():
    Application.set_input_video(r'c:\repos\pattern_movies\texturi_dinamice\54ab110.avi')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.set_output_image_folder('Logs/input_data')

    Application.do_get_video_job(port_output_name='RAW')
    frame_0, frame_1 = Application.do_deep_video_deinterlacing(port_input_name='RAW')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=[frame_1 + '_L0'], job_name_in_port=False)
    Application.run_application()

    Utils.close_files()

def main():
    Application.set_input_image_folder('Logs/input_data/DEEP_DEINTERLACE_FRAME_1_L0')
    Application.set_output_image_folder('Logs/process_data')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')
    Application.do_glcm_job(port_input_name=grey, distance=[1], angles=[0],
                            calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                            calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    Utils.close_files()


if __name__ == "__main__":
    # main_get_img_from_movie()
    main()