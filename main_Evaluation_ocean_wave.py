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
    Application.set_input_video(r'c:\repos\pattern_movies\texturi_dinamice\649f510.avi')

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
    list_to_plot = list()
    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[1], angles=[0],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[1], angles=[np.pi],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[1], angles=[np.pi/4],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[2], angles=[0],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[2], angles=[np.pi],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[2], angles=[np.pi/4],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[4], angles=[0],
                                                calculate_contrast=True, calculate_dissimilarity=True, calculate_homogeneity=True,
                                                calculate_ASM=True, calculate_energy=True, calculate_correlation=True, calculate_entropy=True))

    list_to_plot.append(Application.do_glcm_job(port_input_name=grey, distance=[4], angles=[np.pi],
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

    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='CONTRAST', title='649f510')
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='DISSIMILARITY', title='649f510')
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='HOMOGENEITY', title='649f510')
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='ASM', title='649f510')
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='ENERGY', title='649f510')
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='CORRELATION', title='649f510')
    Utils.plot_GLCM_data(port_list=list_to_plot, caracteristic='ENTROPY', title='649f510')

    Utils.close_files()


if __name__ == "__main__":
    # main_get_img_from_movie()
    main()