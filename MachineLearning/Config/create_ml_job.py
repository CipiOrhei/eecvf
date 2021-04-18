import glob
import os
import shutil

import config_main
from Utils.log_handler import log_ml_info_to_console, log_setup_info_to_console


def set_input_train_image_folder(location_train: str = '', location_label: str = ''):
    """
    Sets the location of image and label data for learning
    :param location_train: location relative to repo
    :param location_label: location relative to repo
    :return: None
    """
    log_setup_info_to_console('INPUT DATA FOR ML TRAIN SETTING UP')
    image_path = os.path.join(os.getcwd(), location_train).replace('\\', '/')
    new_image_path = image_path.replace(image_path.split('/')[-1], 'image')

    config_main.ML_TRAIN_IMG_LOCATION = new_image_path.split('/image')[0]
    log_ml_info_to_console('Train set location set: {}'.format(config_main.ML_TRAIN_IMG_LOCATION))
    os.rename(src=image_path, dst=new_image_path)

    os.mkdir(os.path.join(config_main.ML_TRAIN_IMG_LOCATION) + '/label')

    for dir_name, dir_names, filenames in os.walk(location_label):
        for filename in filenames:
            shutil.copy2(os.path.join(location_label, filename), os.path.join(config_main.ML_TRAIN_IMG_LOCATION, 'label'))


def set_image_input_folder(location_img_in: str = ''):
    """
    Sets the location of training input images for machine learning jobs.
    :param location_img_in: location relative to repo of training images
    :return: None
    """
    config_main.ML_TRAIN_IMG_LOCATION = location_img_in
    log_ml_info_to_console('Training image set location to: {}'.format(config_main.ML_TRAIN_IMG_LOCATION))


def set_label_input_folder(location_img_label_in: str = ''):
    """
    Sets the location of labelled images for machine learning jobs.
    :param location_img_label_in: location relative to repo of training images
    :return: None
    """
    config_main.ML_LABEL_IMG_LOCATION = location_img_label_in
    log_ml_info_to_console('Labelled image set location to: {}'.format(config_main.ML_LABEL_IMG_LOCATION))


def set_image_validate_folder(location_img_in: str = ''):
    """
    Sets the location of validation input images for machine learning jobs.
    :param location_img_in: location relative to repo of training images
    :return: None
    """
    config_main.ML_VALIDATE_IMG_LOCATION = location_img_in
    log_ml_info_to_console('Validate image set location to: {}'.format(config_main.ML_VALIDATE_IMG_LOCATION))


def set_label_validate_folder(location_img_label_in: str = ''):
    """
    Sets the location of validation labelled images for machine learning jobs.
    :param location_img_label_in: location relative to repo of training images
    :return: None
    """
    config_main.ML_LABEL_VALIDATE_LOCATION = location_img_label_in
    log_ml_info_to_console('Labelled validate image set location to: {}'.format(config_main.ML_LABEL_VALIDATE_LOCATION))


def set_output_model_folder(location_out: str = ''):
    """
    Sets the location of output model of training results.
    :param location_out: location relative to repo
    :return: None
    """
    config_main.ML_OUTPUT_IMG_LOCATION = location_out
    log_ml_info_to_console('Model output location set: {}'.format(location_out))


def set_output_checkpoint_location(location_out: str = ''):
    """
    Sets the location of output images if desired
    :param location_out: location relative to repo
    :return: None
    """
    config_main.ML_WEIGHT_OUTPUT_LOCATION = location_out
    log_ml_info_to_console('Checkpoint location set to: {}'.format(location_out))


def clear_model_trained():
    """
    Deletes model from folder
    :return: None
    """
    path = os.path.join(os.getcwd(), config_main.ML_WEIGHT_OUTPUT_LOCATION, '*')
    files = glob.glob(path)

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    log_setup_info_to_console('DELETED CONTENT OF: {}'.format(config_main.ML_WEIGHT_OUTPUT_LOCATION))


def delete_folder_ai_out(folder_name:str = None) -> None:
    """
    Service that deletes the content of out folder where the saved images are.
    :param: folder name if changed
    :return: None
    """
    if folder_name is None:
        path = os.path.join(os.getcwd(), config_main.ML_OUTPUT_IMG_LOCATION, '*')
    else:
        path = os.path.join(os.getcwd(), folder_name, '*')

    files = glob.glob(path)

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    log_setup_info_to_console('DELETED CONTENT OF: {}'.format(config_main.ML_OUTPUT_IMG_LOCATION))


if __name__ == "__main__":
    pass
