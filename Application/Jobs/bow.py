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

def flann_matching(des1,des2, comp_thr = 0.85):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des1 = np.float32(des1)
    des2 = np.float32(des2)

    matches = flann.knnMatch(des1, des2, k=2)
    good = []

    for i, (m, n) in enumerate(matches):
        if m.distance < comp_thr * n.distance:
            good.append([m])
    percent = len(good)/len(des1) * 100

    return percent


class ZuBuD_BOW:

    def __init__(self, name, dict_size, is_from_file=False):
        self.name = name
        self.list_bows = dict()
        self.list_clustered_bows = dict()

        # creates bows for each object object0003.view03.png
        if is_from_file is True:
            for i in range(1, 202):
                name_building_class = 'object{:04d}'.format(i)
                self.list_bows[name_building_class] = None
                self.list_clustered_bows[name_building_class] = None

    def add(self, id, desc, dict_size):
        desc = np.float32(desc)
        if id not in self.list_bows.keys():
            self.list_bows[id] = cv2.BOWKMeansTrainer(dict_size)
            self.list_bows[id].clear()
        self.list_bows[id].add(desc)

    def check(self):
        for key in self.list_bows.keys():
            print(self.list_bows[key].descriptorsCount())

    def cluster_all(self):
        for key in self.list_bows.keys():
            desC = self.list_bows[key].cluster()
            self.list_clustered_bows[key] = desC
            self.list_bows[key].clear()

    def save_to_files_np(self, location):
        for key in self.list_clustered_bows.keys():
            if not os.path.exists(location):
                os.makedirs(location)
            location_np = os.path.join(location, key)
            np.save(location_np, self.list_clustered_bows[key])

    def save_to_files_txt(self, location):
        for key in self.list_clustered_bows.keys():
            if not os.path.exists(location):
                os.makedirs(location)
            location_np = os.path.join(location, key + '.txt')
            np.savetxt(location_np, self.list_clustered_bows[key])

    def populate_from_npy_files(self, location, port):
        for key in self.list_clustered_bows.keys():
            dst_file = os.path.join(location,port,key + '.npy')
            self.list_clustered_bows[key] = np.load(dst_file)




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
object_list_des = dict()
bow_cluster_list = list()

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

        global  object_list_des

        if name not in object_list_des.keys():
            object_list_des[name] = list()

        object_list_des[name].append(port_in.arr)

        # cluster_bow[port_out.name].add(name, np.float32(port_in.arr), param_list[PORT_IN_DESC_SIZE])

        file_to_save = os.path.join(config_main.APPL_SAVE_LOCATION, port_out.name)
        if not os.path.exists(file_to_save):
            os.makedirs(file_to_save)

        location_np = os.path.join(file_to_save, global_var_handler.PICT_NAME[:-4] + '_des.txt')
        np.savetxt(location_np, port_in.arr)


        if global_var_handler.NR_PICTURES - 1 == global_var_handler.FRAME:

            for key in object_list_des.keys():
                BOW = cv2.BOWKMeansTrainer(param_list[PORT_IN_DESC_SIZE])

                for el in object_list_des[key]:
                    BOW.add(np.float32(el))

                desC = BOW.cluster()
                bow_cluster_list.append(desC)
                # only in last wave
                BOW.clear()

            keys = list(object_list_des.keys())
            for idx in range(len(bow_cluster_list)):
                cluster_bow[port_out.name].list_clustered_bows[keys[idx]] = bow_cluster_list[idx]

            file_to_save = os.path.join(config_main.APPL_SAVE_LOCATION, port_out.name)

            if param_list[PORT_IN_SAVE_TXT]:
                cluster_bow[port_out.name].save_to_files_txt(file_to_save)

            if param_list[PORT_IN_SAVE_NPY]:
                cluster_bow[port_out.name].save_to_files_np(file_to_save)

            # t = cluster_bow[port_out.name].list_clustered_bows

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


def bow_zubud_flann_inquiry_main_func(param_list: list = None) -> bool:
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
    PORT_IN_SAVED_TXT = 2
    # noinspection PyPep8Naming
    PORT_IN_SAVED_NPY = 3
    # noinspection PyPep8Naming
    PORT_IN_FLANN_THR = 4
    # noinspection PyPep8Naming
    PORT_IN_LOCATION_BOW = 5
    # noinspection PyPep8Naming
    PORT_IN_PORT_BOW = 6
    # noinspection PyPep8Naming
    PORT_OUT = 7
    # verify that the number of parameters are OK.
    if len(param_list) != 8:
        log_error_to_console("ZuBuD_BOW_FLANN_INQUIRY JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])

        global cluster_bow

        port_out = get_port_from_wave(name=param_list[PORT_OUT])

        # CHANGE IF NAME OF BOW CHANGES
        size_dict = int(param_list[PORT_IN_PORT_BOW][10:13])

        cluster_bow[param_list[PORT_IN_PORT_BOW]] = ZuBuD_BOW(name=param_list[PORT_IN_PORT_BOW], dict_size=size_dict, is_from_file=True)

        if param_list[PORT_IN_SAVED_NPY]:
            cluster_bow[param_list[PORT_IN_PORT_BOW]].populate_from_npy_files(location=param_list[PORT_IN_LOCATION_BOW], port=param_list[PORT_IN_PORT_BOW])

        percent_class_dict = dict()
        for key in cluster_bow[param_list[PORT_IN_PORT_BOW]].list_clustered_bows.keys():
            try:
                percent_class_dict[key] = flann_matching(des1=port_in.arr,
                                                         des2=cluster_bow[param_list[PORT_IN_PORT_BOW]].list_clustered_bows[key],
                                                         comp_thr=param_list[PORT_IN_FLANN_THR])
            except:
                percent_class_dict[key] = 0

        percent_class_dict = dict(sorted(percent_class_dict.items(), key=lambda item: item[1], reverse=True))
        # t = np.array(percent_class_dict.items())
        t = np.array([[int(a[-4:]), float(x)] for a, x in percent_class_dict.items()])
        port_out.arr[:] = t

        file_to_save = os.path.join(config_main.APPL_SAVE_LOCATION, port_out.name)
        if not os.path.exists(file_to_save):
            os.makedirs(file_to_save)
        location_np = os.path.join(file_to_save, global_var_handler.PICT_NAME.split('.')[0] + '.txt')
        np.savetxt(fname=location_np, X=port_out.arr, fmt=['%3.0f', '%3.3f'])

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


def do_zubud_bow_inquiry_flann_job(port_to_inquiry: str, flann_thr: float, location_of_bow: str, bow_port: str,
                                   saved_to_text: bool = False, saved_to_npy: bool = True,
                                   port_out_name: str = None,
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
    input_port_name = transform_port_name_lvl(name=port_to_inquiry, lvl=level)

    if port_out_name is None:
        port_out_name = 'ZuBuD_BOW_INQ_THR_{thr}_{Input}'.format(thr=flann_thr.__str__().replace('.', '_'), Input=bow_port[0:-3])

    port_img_output_name = transform_port_name_lvl(name=port_out_name, lvl=level)
    port_des_output_name_size = '(201, 2)'

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, saved_to_text, saved_to_npy, flann_thr, location_of_bow, bow_port, port_img_output_name]
    output_port_list = [(port_img_output_name, port_des_output_name_size, 'f', False)]

    job_name = job_name_create(action='ZuBuD_BOW inquiry FLANN {thr}'.format(thr=flann_thr.__str__().replace('.','_')),
                               input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='bow_zubud_flann_inquiry_main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_out_name



if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
