# import what you need
import config_main
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file


import cv2
import numpy as np
import os

"""
Module handles DESCRIPTION OF THE MODULE jobs for the APPL block.
"""

############################################################################################################################################
# Internal functions
############################################################################################################################################

cluster_bow = dict()

class ZuBuD_BOW:

    def __init__(self, name, dict_size):
        self.name = name
        self.list_bows = dict()
        self.list_clustered_bows = dict()
        # creates bows for each object object0003.view03.png
        for i in range(1, 202):
            name_building_class = 'object{:04d}'.format(i)
            self.list_bows[name_building_class] = cv2.BOWKMeansTrainer(dict_size)

    def add(self, id, desc):
        desc = np.float32(desc)
        self.list_bows[id].add(desc)

    def check(self):
        for key in self.list_bows.keys():
            print(self.list_bows[key].descriptorsCount())

    def cluster_all(self):
        for key in self.list_bows.keys():
            self.list_clustered_bows[key] = self.list_bows[key].cluster()

    def save_to_files_np(self, location):
        for key in self.list_bows.keys():
            if not os.path.exists(location):
                os.makedirs(location)
            location_np = os.path.join(location, key)
            np.save(location_np, self.list_clustered_bows[key])

    def save_to_files_txt(self, location):
        for key in self.list_bows.keys():
            if not os.path.exists(location):
                os.makedirs(location)
            location_np = os.path.join(location, key + '.txt')
            np.savetxt(location_np, self.list_clustered_bows[key])




############################################################################################################################################
# Init functions
############################################################################################################################################

# define a init function, function that will be executed at the begging of the wave
def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    # log_to_file('DATA YOU NEED TO SAVE EVERY FRAME IN CSV')
    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################


def bow_zubud_main_func(param_list: list = None) -> bool:
    """
    Main function for ZuBuD BOW calculation job.
    :param param_list: Param needed to respect the following list:
                       [port to add, wave of input port, size of cluster, save to txt, save to npy, output ]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_DESC_SIZE = 2
    # noinspection PyPep8Naming
    PORT_IN_SAVE_NPY = 3
    # noinspection PyPep8Naming
    PORT_IN_SAVE_TXT = 4
    # noinspection PyPep8Naming
    PORT_OUT_BOW = 5
    # verify that the number of parameters are OK.
    if len(param_list) != 6:
        log_error_to_console("ZuBuD_BOW JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])

        global cluster_bow

        port_out = get_port_from_wave(name=param_list[PORT_OUT_BOW])

        if port_out.name not in cluster_bow.keys():
            cluster_bow[port_out.name] = ZuBuD_BOW(name=port_in.name, dict_size=param_list[PORT_IN_DESC_SIZE])

        name = global_var_handler.PICT_NAME[:10]
        cluster_bow[port_out.name].add(name, port_in.arr)

        if global_var_handler.NR_PICTURES - 1 == global_var_handler.FRAME:
            # only in last wave
            cluster_bow[port_out.name].cluster_all()
            file_to_save = os.path.join(config_main.APPL_SAVE_LOCATION, port_out.name)

            if param_list[PORT_IN_SAVE_TXT]:
                cluster_bow[port_out.name].save_to_files_txt(file_to_save)

            if param_list[PORT_IN_SAVE_NPY]:
                cluster_bow[port_out.name].save_to_files_np(file_to_save)

            t = cluster_bow[port_out.name].list_clustered_bows

            t = np.array(list(cluster_bow[port_out.name].list_clustered_bows.values()))
            port_out.arr[:] = t
            port_out.set_valid()

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                pass
            except BaseException as error:
                log_error_to_console("ZuBuD_BOW JOB NOK: ", str(error))
                pass
        else:
            return False

        return True

############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_zubud_bow_job(port_to_add: str, dictionary_size: int,
                     save_to_text: bool = True, save_to_npy: bool = True,
                     port_bow_list_output: str = None,
                     level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Bag  of  Features  (BOF)  has  become  a  popular  approach in  CV  for  image  classification,  object  detection,  or  image retrieval.
    The  name  comes  from  the  Bag  of  Words  (BOW)representation used in textual information retrieval
    # https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/csurka-eccv-04.pdf
    :param port_to_add:  Feature port to add to BOW
    :param dictionary_size: size of cluster per class
    :param save_to_text: if we want to save BOW clusters to txt files
    :param save_to_npy: if we want to save BOW clusters to npy files
    :param port_bow_list_output: output port with BOW
    :param level: pyramid level to calculate at
    :param wave_offset: wave of input port, please correlate with each input port name parameter
    :return: Name of output port or ports
    """
    # Do this for each input port this function has
    input_port_name = transform_port_name_lvl(name=port_to_add, lvl=level)

    if port_bow_list_output is None:
        port_output = '{name}_{size}_{Input}'.format(name='ZuBuD_BOW', size=dictionary_size.__str__(), Input=port_to_add)

    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_output, lvl=level)
    port_des_output_name_size = '(201, {size}, 64)'.format(size=dictionary_size)
    # port_des_output_name_size = '(201, 2)'.format(size=dictionary_size)
    # port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, dictionary_size, save_to_text, save_to_npy, port_img_output_name]
    output_port_list = [(port_img_output_name, port_des_output_name_size, 'f', False)]

    job_name = job_name_create(action='ZuBuD_BOW {size}'.format(size=dictionary_size.__str__()),
                               input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='bow_zubud_main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
