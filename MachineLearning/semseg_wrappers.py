from MachineLearning.external.image_segmentation_keras.keras_segmentation.train import train
from MachineLearning.Config.create_ml_job import set_output_checkpoint_location
import config_main as CONFIG
import os

from Utils.log_handler import log_error_to_console

"""
Module for training details of semseg models
"""


def do_semseg_base(model: str, input_height=None, input_width=None, n_classes=None, epochs=5, verify_dataset=True,
                   steps_per_epoch=512, val_steps_per_epoch=512, batch_size=2, optimizer_name='adam') -> None:
    """
    TODO detail this
    This uses the external repo: https://github.com/divamgupta/image-segmentation-keras

    :param model: model to use for training
                    ["fcn_8", "fcn_32", "fcn_8_vgg", "fcn_32_vgg", "fcn_8_resnet50", "fcn_32_resnet50", "fcn_8_mobilenet",
                     "fcn_32_mobilenet", "pspnet", "vgg_pspnet", "resnet50_pspnet", "vgg_pspnet", "resnet50_pspnet", "pspnet_50",
                     "pspnet_101", "unet_mini", "unet", "vgg_unet", "resnet50_unet", "mobilenet_unet", "segnet", "vgg_segnet",
                     "resnet50_segnet", "mobilenet_segnet" ]
    :param verify_dataset: if we want to validate the input data
    :param input_height: height of input training images
    :param input_width: width of input training images
    :param n_classes: number of classes
    :param epochs: epochs to use in training
    :param batch_size: determines the number of samples in each mini batch.
    :param steps_per_epoch: the number of batch iterations before a training epoch is considered finished.
    :param val_steps_per_epoch: similar to steps_per_epoch but on the validation data set instead on the training data.
    :param optimizer_name: name of the optimizer
    :return: None
    """
    model_list = ["fcn_8", "fcn_32", "fcn_8_vgg", "fcn_32_vgg", "fcn_8_resnet50", "fcn_32_resnet50", "fcn_8_mobilenet", "fcn_32_mobilenet",
                  "pspnet", "vgg_pspnet", "resnet50_pspnet", "vgg_pspnet", "resnet50_pspnet", "pspnet_50", "pspnet_101", "unet_mini",
                  "unet", "vgg_unet", "resnet50_unet", "mobilenet_unet", "segnet", "vgg_segnet", "resnet50_segnet", "mobilenet_segnet"]
    model_name = None

    if model in model_list:
        model_name = model

    set_output_checkpoint_location(os.path.join('MachineLearning/model_weights', model_name))

    if not os.path.exists(CONFIG.ML_WEIGHT_OUTPUT_LOCATION):
        os.makedirs(CONFIG.ML_WEIGHT_OUTPUT_LOCATION)

    try:
        # TODO add all fields
        train(model=model_name,
              train_images=CONFIG.ML_TRAIN_IMG_LOCATION,
              train_annotations=CONFIG.ML_LABEL_IMG_LOCATION, validate=True,
              val_images=CONFIG.ML_VALIDATE_IMG_LOCATION, val_annotations=CONFIG.ML_LABEL_VALIDATE_LOCATION,
              # val_images=None, val_annotations=None,
              epochs=epochs, n_classes=n_classes, input_height=input_height, input_width=input_width, batch_size=batch_size,
              verify_dataset=verify_dataset, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch,
              checkpoints_path=CONFIG.ML_WEIGHT_OUTPUT_LOCATION + '/' + model_name,
              # checkpoints_path=CONFIG.ML_WEIGHT_OUTPUT_LOCATION,
              optimizer_name=optimizer_name,
              auto_resume_checkpoint=True,
              gen_use_multiprocessing=False)

    except BaseException as error:
        log_error_to_console("SEMSEG " + model_name.upper() + " TRAIN JOB NOK: ", str(error))
        return


def do_vgg_unet_training(input_height=None, input_width=None, n_classes=None, epochs=5, verify_dataset=True, batch_size=2,
                         steps_per_epoch=512, val_steps_per_epoch=512, optimizer_name='adam', ):
    """
    TODO detail this

    This uses the external repo: https://github.com/divamgupta/image-segmentation-keras
    Paper:     https://arxiv.org/abs/1411.4038
    :param verify_dataset: if we want to validate the input data
    :param input_height: height of input training images
    :param input_width: width of input training images
    :param n_classes: number of classes
    :param epochs: epochs to use in training
    :param batch_size: determines the number of samples in each mini batch.
    :param steps_per_epoch: the number of batch iterations before a training epoch is considered finished.
    :param val_steps_per_epoch: similar to steps_per_epoch but on the validation data set instead on the training data.
    :param optimizer_name: name of the optimizer
    :return: None
    """

    do_semseg_base(model="vgg_unet", input_height=input_height, input_width=input_width, n_classes=n_classes, epochs=epochs,
                   verify_dataset=verify_dataset, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch,
                   optimizer_name=optimizer_name, batch_size=batch_size)


def do_unet_training(input_height=None, input_width=None, n_classes=None, epochs=5, verify_dataset=True, batch_size=2,
                     steps_per_epoch=512, val_steps_per_epoch=512, optimizer_name='adam', ):
    """
    TODO detail this

    This uses the external repo: https://github.com/divamgupta/image-segmentation-keras
    Paper:     https://arxiv.org/abs/1505.04597
    :param verify_dataset: if we want to validate the input data
    :param input_height: height of input training images
    :param input_width: width of input training images
    :param n_classes: number of classes
    :param epochs: epochs to use in training
    :param batch_size: determines the number of samples in each mini batch.
    :param steps_per_epoch: the number of batch iterations before a training epoch is considered finished.
    :param val_steps_per_epoch: similar to steps_per_epoch but on the validation data set instead on the training data.
    :param optimizer_name: name of the optimizer
    :return: None
    """

    do_semseg_base(model="unet", input_height=input_height, input_width=input_width, n_classes=n_classes, epochs=epochs,
                   verify_dataset=verify_dataset, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch,
                   optimizer_name=optimizer_name, batch_size=batch_size)


def do_resnet50_unet_training(input_height=None, input_width=None, n_classes=None, epochs=5, verify_dataset=True, batch_size=2,
                              steps_per_epoch=512, val_steps_per_epoch=512, optimizer_name='adam', ):
    """
    TODO detail this

    This uses the external repo: https://github.com/divamgupta/image-segmentation-keras
    Paper:     TODO add paper
    :param verify_dataset: if we want to validate the input data
    :param input_height: height of input training images
    :param input_width: width of input training images
    :param n_classes: number of classes
    :param epochs: epochs to use in training
    :param batch_size: determines the number of samples in each mini batch.
    :param steps_per_epoch: the number of batch iterations before a training epoch is considered finished.
    :param val_steps_per_epoch: similar to steps_per_epoch but on the validation data set instead on the training data.
    :param optimizer_name: name of the optimizer
    :return: None
    """

    do_semseg_base(model="resnet50_unet", input_height=input_height, input_width=input_width, n_classes=n_classes, epochs=epochs,
                   verify_dataset=verify_dataset, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch,
                   optimizer_name=optimizer_name, batch_size=batch_size)


def do_mobilenet_unet_training(n_classes=None, epochs=5, verify_dataset=True, batch_size=2,
                               steps_per_epoch=512, val_steps_per_epoch=512, optimizer_name='adam', ):
    """
    TODO detail this
    Mobile net works only on 224 x 224
    This uses the external repo: https://github.com/divamgupta/image-segmentation-keras
    Paper:     TODO add paper
    :param verify_dataset: if we want to validate the input data
    :param input_height: height of input training images
    :param input_width: width of input training images
    :param n_classes: number of classes
    :param epochs: epochs to use in training
    :param batch_size: determines the number of samples in each mini batch.
    :param steps_per_epoch: the number of batch iterations before a training epoch is considered finished.
    :param val_steps_per_epoch: similar to steps_per_epoch but on the validation data set instead on the training data.
    :param optimizer_name: name of the optimizer
    :return: None
    """

    do_semseg_base(model="mobilenet_unet", input_height=224, input_width=224, n_classes=n_classes, epochs=epochs,
                   verify_dataset=verify_dataset, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch,
                   optimizer_name=optimizer_name, batch_size=batch_size)


def do_unet_mini_training(input_height=None, input_width=None, n_classes=None, epochs=5, verify_dataset=True, batch_size=2,
                          steps_per_epoch=512, val_steps_per_epoch=512, optimizer_name='adam', ):
    """
    TODO detail this

    This uses the external repo: https://github.com/divamgupta/image-segmentation-keras
    Paper:     TODO add paper
    :param verify_dataset: if we want to validate the input data
    :param input_height: height of input training images
    :param input_width: width of input training images
    :param n_classes: number of classes
    :param epochs: epochs to use in training
    :param batch_size: determines the number of samples in each mini batch.
    :param steps_per_epoch: the number of batch iterations before a training epoch is considered finished.
    :param val_steps_per_epoch: similar to steps_per_epoch but on the validation data set instead on the training data.
    :param optimizer_name: name of the optimizer
    :return: None
    """

    do_semseg_base(model="unet_mini", input_height=input_height, input_width=input_width, n_classes=n_classes, epochs=epochs,
                   verify_dataset=verify_dataset, steps_per_epoch=steps_per_epoch, val_steps_per_epoch=val_steps_per_epoch,
                   optimizer_name=optimizer_name, batch_size=batch_size)


if __name__ == "__main__":
    pass
