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


def main_get_img_from_movie(video):
    Application.set_input_video(video)
    Application.set_output_image_folder('Logs/input_data')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_video_job(port_output_name='RAW')
    frame_0, frame_1 = Application.do_deep_video_deinterlacing(port_input_name='RAW')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=[frame_1 + '_L0'], job_name_in_port=False)
    Application.run_application()

    Utils.close_files()


def create_other_directions():
    Application.set_input_image_folder('Logs/input_data/DEEP_DEINTERLACE_FRAME_1_L0')
    Application.set_output_image_folder('Logs/process_cube')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')
    Application.create_image_cube(port_input_name='RAW', is_rgb=True, location_to_save='Logs/process_cube')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()
    Utils.close_files()


def main(t, input, title):
    Application.set_input_image_folder(input)
    Application.set_output_image_folder('Logs/process_data')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')
    list_to_plot = list()
    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[1], angles=[0],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[1], angles=[np.pi/2],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[1], angles=[np.pi/4],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[2], angles=[0],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[2], angles=[np.pi/2],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[2], angles=[np.pi/4],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[4], angles=[0],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[4], angles=[np.pi/2],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[4], angles=[np.pi / 4],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=False)
    Application.run_application()

    for el in range(len(list_to_plot)):
        list_to_plot[el] += '_LC0'

    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='CONTRAST', title=title, table_number=t)
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='DISSIMILARITY', title=title, table_number=t)
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='HOMOGENEITY', title=title, table_number=t)
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='ASM', title=title, table_number=t)
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='ENERGY', title=title, table_number=t)
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='CORRELATION', title=title, table_number=t)
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='ENTROPY', title=title, table_number=t)

    Utils.close_files()


if __name__ == "__main__":
    main_get_img_from_movie(video=r'c:\repos\pattern_movies\texturi_dinamice\649f510.avi')
    Utils.reopen_files()
    create_other_directions()
    Utils.reopen_files()
    # if we run the movie->img transformation set t=2 else t=1
    main(t=3, input='Logs/input_data/DEEP_DEINTERLACE_FRAME_1_L0', title='649f510_front_to_back')
    Utils.reopen_files()
    main(t=4, input='Logs/process_cube/COLUMN_SLICING_IMG_CUBE_RAW_L0', title='649f510_left_to_right')
    Utils.reopen_files()
    main(t=5, input='Logs/process_cube/LINE_SLICING_IMG_CUBE_RAW_L0', title='649f510_top_to_botton')