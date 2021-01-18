import config_main

from Application.Schedulers.simple_RR import run_rr
from Application.Frame.transferJobPorts import prepare_ports_new_wave, log_to_console_exchange_ports
from Utils.log_handler import log_to_console, log_setup_info_to_console
from Application.Utils.TimeLogger import Timer
from Application.Utils.parseInputFolder import get_picture_size_and_number, get_video_capture, get_camera_capture, release_video, \
    clear_input_img_dir
from Application.Utils.parseJsonFile import get_jobs
from Application.Utils.image_handler import show_pictures, save_pict_to_file
from Application.Frame.job_handler import job_creation, init_jobs, log_to_console_avg_time, terminate_jobs
from Application.Frame.global_variables import global_var_handler


def run_application():
    """
    Main function of system
    :return: None
    """
    # initialize timers
    timer_setup = Timer()
    timer_init = Timer()
    timer_application = Timer()
    timer_wave = Timer()
    timer_post_processing = Timer()

    if config_main.APPL_INPUT is not None:
        log_setup_info_to_console("SETUP STEP")
        timer_setup.start_cycle_timer()
        global_var_handler()
        if config_main.APPL_INPUT == config_main.IMAGE_INPUT:
            get_picture_size_and_number()
        elif config_main.APPL_INPUT == config_main.VIDEO_INPUT:
            get_video_capture()
        elif config_main.APPL_INPUT == config_main.CAMERA_INPUT:
            get_camera_capture()
        log_setup_info_to_console("JOB CREATION STEP")
        job_list = job_creation(job_description=get_jobs(json_file=config_main.APPL_INPUT_JOB_LIST))
        timer_setup.end_cycle_timer()
        timer_setup.cycle_updater()
        log_setup_info_to_console("JOB INIT STEP")
        timer_init.start_cycle_timer()
        init_jobs(list_jobs=job_list)
        timer_init.end_cycle_timer()
        timer_init.cycle_updater()
        log_setup_info_to_console("JOB RUN STEP")
        timer_application.start_cycle_timer()
        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        while global_var_handler.FRAME < global_var_handler.NR_PICTURES:
            timer_wave.start_cycle_timer()
            # noinspection PyUnresolvedReferences
            log_to_console('FRAME {}'.format(global_var_handler.FRAME))
            run_rr(jobs=job_list)
            timer_wave.end_cycle_timer()
            timer_wave.cycle_updater()
            timer_post_processing.start_cycle_timer()
            show_pictures()
            save_pict_to_file()
            timer_post_processing.end_cycle_timer()
            timer_post_processing.cycle_updater()
            # noinspection PyUnresolvedReferences
            global_var_handler.FRAME += 1
            # noinspection PyUnresolvedReferences
            prepare_ports_new_wave(frame=global_var_handler.FRAME)
        timer_application.end_cycle_timer()
        timer_application.cycle_updater()
        log_setup_info_to_console("TERMINATE STEP")
        terminate_jobs(job_list)

        if config_main.APPL_INPUT == config_main.VIDEO_INPUT or config_main.APPL_INPUT == config_main.CAMERA_INPUT:
            release_video()
        else:
            clear_input_img_dir()

        log_setup_info_to_console("PHASE SETUP AVERAGE TIME[s]            : {time:10.10f}".format(time=timer_setup.__average_time_sum__))
        log_setup_info_to_console("PHASE INIT AVERAGE TIME[s]             : {time:10.10f}".format(time=timer_init.__average_time_sum__))
        log_setup_info_to_console(
            "PHASE WAVE AVERAGE TIME[s]             : {time:10.10f}".format(time=timer_wave.get_average_time_seconds()))
        log_setup_info_to_console(
            "PHASE POST PROCESSING AVERAGE TIME[s]  : {time:10.10f}".format(time=timer_post_processing.get_average_time_seconds()))
        log_setup_info_to_console(
            "PHASE RUN AVERAGE TIME[s]              : {time:10.10f}".format(time=timer_application.__average_time_sum__))

        log_to_console_avg_time(job_list)
        log_to_console_exchange_ports()
    else:
        log_setup_info_to_console('NO INPUT FOR APPLICATION')


if __name__ == "__main__":
    config_main.APPL_INPUT_DIR = '../' + config_main.APPL_INPUT_DIR
    config_main.APPL_INPUT_JOB_LIST = config_main.APPL_INPUT_JOB_LIST.replace('Application/', '')
    config_main.APPL_SAVE_LOCATION = '../' + config_main.APPL_SAVE_LOCATION
    run_application()
