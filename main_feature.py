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
Code for paper:
  title={CBIR for urban building using A-KAZE features},
  author={Orhei, Ciprian and Radu, Lucian and Mocofan, Muguras and Vert, Silviu and Vasiu, Radu},
  booktitle={2021 IEEE 27nd International Symposium for Design and Technology in Electronic Packaging (SIITME)},
  pages={--},
  year={2021},
  organization={IEEE}
"""
# Original code in : https://github.com/Tatsu21/feature-detection

def main_bow_create():
    """
    Main function of framework Please look in example_main for all functions you can use
    """
    Application.set_input_image_folder(r'c:/repos/ZuBud_dataset/png-ZuBuD')
    Application.delete_folder_appl_out()

    grey = Application.do_get_image_job(port_output_name='GRAY_RAW', direct_grey=True)
    # grey = Application.do_grayscale_transform_job(port_input_name='RAW')

    # desc_list = [cv2.AKAZE_DESCRIPTOR_KAZE, cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT, cv2.AKAZE_DESCRIPTOR_MLDB, cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT]
    desc_list = [cv2.AKAZE_DESCRIPTOR_KAZE]
    # diff_list = [cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2, cv2.KAZE_DIFF_CHARBONNIER, cv2.KAZE_DIFF_WEICKERT]
    diff_list = [cv2.KAZE_DIFF_PM_G1]
    # desc_size_list = [0, 8, 16, 32, 64, 128]
    desc_size_list = [64]
    nOctaves_list = [5]
    nLayes_list = [6]
    thr_list = [0.85]
    # thr_akaze_list = [0.0010, 0.0011, 0.0012, 0.0013]
    thr_akaze_list = [0.0012]
    # dictionarySize_list = [375, 400, 425]
    dictionarySize_list = [400]

    list_to_eval = list()

    for desc in desc_list:
        for diff in diff_list:
            for desc_size in desc_size_list:
                for nOctaves in nOctaves_list:
                    for nLayes in nLayes_list:
                        for thr in thr_list:
                            for thr_akaze in thr_akaze_list:
                                for dict_size in dictionarySize_list:
                                    kp, des, img = Application.do_a_kaze_job(port_input_name=grey, descriptor_channels=1,
                                                                             descriptor_size=desc_size, descriptor_type=desc, diffusivity=diff,
                                                                             threshold=thr_akaze, nr_octaves=nOctaves, nr_octave_layers=nLayes)

                                    bow = Application.do_zubud_bow_job(port_to_add=des, dictionary_size=dict_size)

    Application.create_config_file()
    # Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save='ALL')
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save=[])
    Application.run_application()

    Application.set_input_image_folder(r'c:/repos/ZuBud_dataset/qimage')
    Application.set_output_image_folder('Logs/query_application')
    Application.delete_folder_appl_out()

    grey = Application.do_get_image_job(port_output_name='GRAY_RAW', direct_grey=True)
    # grey = Application.do_grayscale_transform_job(port_input_name='RAW')

    for desc in desc_list:
        for diff in diff_list:
            for desc_size in desc_size_list:
                for nOctaves in nOctaves_list:
                    for nLayes in nLayes_list:
                        for thr in thr_list:
                            for thr_akaze in thr_akaze_list:
                                for dict_size in dictionarySize_list:
                                    kp, des, img = Application.do_a_kaze_job(port_input_name=grey, descriptor_channels=1,
                                                                             descriptor_size=desc_size, descriptor_type=desc, diffusivity=diff,
                                                                             threshold=thr_akaze, nr_octaves=nOctaves, nr_octave_layers=nLayes)

                                    final = Application.do_zubud_bow_inquiry_flann_job(port_to_inquiry=des, flann_thr=thr, saved_to_npy=True,
                                                                                       location_of_bow='Logs/application_results',
                                                                                       bow_port='ZuBuD_BOW_' + dict_size.__str__() + '_' + des + '_L0')

                                    list_to_eval.append(final + '_L0')

    Application.create_config_file()
    # Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save='ALL')
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save=[])
    Application.run_application()

    Benchmarking.run_CBIR_ZuBuD_benchmark(input_location='Logs/query_application/',
                                          gt_location=r'c:\repos\ZuBud_dataset\zubud_groundtruth.txt',
                                          raw_image=r'c:/repos/ZuBud_dataset/qimage',
                                          jobs_set=list_to_eval)

    Utils.close_files()


if __name__ == "__main__":
    main_bow_create()
