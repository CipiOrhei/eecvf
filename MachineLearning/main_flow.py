from MachineLearning.Models.unet_edge import unet
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
import config_main as CONFIG
from Utils.log_handler import log_ml_info_to_console

gpus = tf.config.experimental.list_physical_devices('GPU')
# noinspection PyUnresolvedReferences
tf.config.experimental.set_memory_growth(gpus[0], True)


def train_generator(batch_size, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale",
                    flag_multi_class=False, num_class=2, target_size=(512, 512), seed=1):
    image_data_gen = ImageDataGenerator(**aug_dict)
    mask_data_gen = ImageDataGenerator(**aug_dict)

    input_path = os.path.join(os.getcwd(), CONFIG.ML_TRAIN_IMG_LOCATION)
    # output_path = os.path.join(os.getcwd(), CONFIG.ML_OUTPUT_IMG_LOCATION)

    image_generator = image_data_gen.flow_from_directory(
        directory=input_path,
        classes=['image'],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    mask_generator = mask_data_gen.flow_from_directory(
        directory=input_path,
        classes=['label'],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
        yield img, mask


def adjust_data(img, mask, flag_multi_class, num_class):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask


def test_generator(test_path, num_image=0, target_size=(512, 512), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        print(np.shape(img))
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img


def do_U_net_model(steps_per_epoch, epochs):
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    my_gene = train_generator(2, data_gen_args)

    if not os.path.isfile(os.path.join(CONFIG.ML_WEIGHT_OUTPUT_LOCATION, 'unet_edge.hdf5')):
        model = unet()
        # add path
        model_checkpoint = ModelCheckpoint(filepath=os.path.join(CONFIG.ML_WEIGHT_OUTPUT_LOCATION, 'unet_edge.hdf5'),
                                           monitor='loss', verbose=0, save_best_only=True)

        model.fit_generator(my_gene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint])
    else:
        log_ml_info_to_console('Model already trained: unet_edge.hdf5')

