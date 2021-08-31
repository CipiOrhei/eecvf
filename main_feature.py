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
Code for paper:
  title={CBIR for urban building using A-KAZE features},
  author={Orhei, Ciprian and Radu, Lucian and Mocofan, Muguras and Vert, Silviu and Vasiu, Radu},
  booktitle={2021 IEEE 27nd International Symposium for Design and Technology in Electronic Packaging (SIITME)},
  pages={--},
  year={2021},
  organization={IEEE}
"""


def main():
    """
    Main function of framework Please look in example_main for all functions you can use
    """
    Application.set_input_image_folder('TestData/smoke_test')
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW')
    Application.do_pyramid_level_down_job(port_input_name='GRAY_RAW', number_of_lvl=1, port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0)
    Application.do_pyramid_level_down_job(port_input_name='RAW', number_of_lvl=1, is_rgb=True, port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0)

    thr_image = Application.do_image_threshold_job(port_input_name='GRAY_RAW', input_value=150,
                                                   input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY, level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

    Application.do_sift_job(port_input_name='GRAY_RAW', mask_port_name=thr_image, level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
    Application.do_sift_job(port_input_name='GRAY_RAW', mask_port_name=None, level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
    Application.do_sift_job(port_input_name='GRAY_RAW', level=CONFIG.PYRAMID_LEVEL.LEVEL_1)


    Application.create_config_file()
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save='ALL')
    Application.run_application()
    Utils.close_files()


if __name__ == "__main__":
    main()
