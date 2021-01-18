from config_main import CUDA_GPU

if CUDA_GPU:
    from MachineLearning.Config.create_ml_job import set_input_train_image_folder
    from MachineLearning.Config.create_ml_job import set_image_input_folder
    from MachineLearning.Config.create_ml_job import set_label_input_folder
    from MachineLearning.Config.create_ml_job import set_output_model_folder
    from MachineLearning.Config.create_ml_job import set_output_checkpoint_location
    from MachineLearning.Config.create_ml_job import clear_model_trained

    from MachineLearning.main_flow import do_U_net_model
    from MachineLearning.semseg_wrappers import do_mobilenet_unet_training
    from MachineLearning.semseg_wrappers import do_unet_mini_training
    from MachineLearning.semseg_wrappers import do_resnet50_unet_training
    from MachineLearning.semseg_wrappers import do_unet_training
    from MachineLearning.semseg_wrappers import do_vgg_unet_training
    from MachineLearning.semseg_wrappers import do_semseg_base
