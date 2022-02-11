# import what you need
import config_main
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file

from Application.Utils.misc import return_kp_cv2_object

import geopy.distance

import cv2
import numpy as np
import os
import json

"""
Module handles DESCRIPTION OF THE MODULE jobs for the APPL block.
"""

############################################################################################################################################
# Internal functions
############################################################################################################################################

cluster_bow = dict()

frame_detection_tracker = dict()


def flann_matching(des1, des2, comp_thr=0.85):
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    index_params = dict(algorithm=4, trees=8)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des1 = np.float32(des1)
    des2 = np.float32(des2)

    matches = flann.knnMatch(des1, des2, k=2)
    good = []

    for i, (m, n) in enumerate(matches):
        if m.distance < comp_thr * n.distance:
            good.append(m)
    percent = len(good) / len(des1) * 100

    return good, percent


class frame_tracker:
    """
    Class for frame tracker
    """

    def __init__(self):
        self.frame_tracker = dict()
        self.frame_decay = 25
        self.frame_boost = 5
        self.threshold_value = 75

    def add_to_frame_tracker(self, obj):
        if obj not in self.frame_tracker:
            self.frame_tracker[obj] = self.frame_boost
        else:
            dict_keys = list(self.frame_tracker.keys())
            self.frame_tracker[obj] += self.frame_boost
            self.frame_tracker[obj] = min(100, self.frame_tracker[obj])

            dict_keys.remove(obj)
            for key in dict_keys:
                self.frame_tracker[key] -= self.frame_decay

    def decay_frame_tracker(self):
        list_to_delete = list()
        dict_keys = list(self.frame_tracker.keys())
        for key in dict_keys:
            # self.frame_tracker[key] -= self.frame_decay
            if self.frame_tracker[key] <= 0:
                list_to_delete.append(key)

        for key in list_to_delete:
            self.frame_tracker.pop(key)

    def decay_all(self):
        list_to_delete = list()
        dict_keys = list(self.frame_tracker.keys())
        for key in dict_keys:
            self.frame_tracker[key] -= self.frame_decay
            self.frame_tracker[key] = max(0, self.frame_tracker[key])
        #     if self.frame_tracker[key] <= 0:
        #         list_to_delete.append(key)
        #
        # for key in list_to_delete:
        #     self.frame_tracker.pop(key)

    def get_frame_max_value(self):
        dict_keys = list(self.frame_tracker.keys())
        if len(dict_keys) != 0:
            t = max(self.frame_tracker, key=self.frame_tracker.get)
            text_to_show = 'Confidence: {} Object ID: {}'.format(self.frame_tracker[t], t)

            if self.frame_tracker[t] >= self.threshold_value:
                return t, text_to_show
            else:
                return None, text_to_show
        else:
            text_to_show = 'VALUE: None CLASS: None'
            return None, text_to_show


class ZuBuD_BOW:
    def __init__(self, name, dict_size, number_classes, is_from_file=False):
        """
        Class for describing the BOF instances
        :param name: name of instance
        :param dict_size: size of cluster
        :param number_classes: number of clusters
        :param is_from_file: if we take it from a saved file
        """
        self.name = name
        self.list_bows = dict()
        self.list_clustered_bows = dict()
        self.list_gps_location = dict()

        # creates bows for each object object0003.view03.png
        if is_from_file is True:
            for i in range(1, number_classes + 1):
                name_building_class = 'object{:04d}'.format(i)
                self.list_bows[name_building_class] = None
                self.list_clustered_bows[name_building_class] = None

    def save_to_files_np(self, location):
        """
        Save clustered BOW to npy file
        :param location: location of file
        :return: None
        """
        for key in self.list_clustered_bows.keys():
            if not os.path.exists(location):
                os.makedirs(location)
            location_np = os.path.join(location, key)
            np.save(location_np, self.list_clustered_bows[key])

    def save_to_files_txt(self, location):
        """
        Save clustered BOW to txt file
        :param location:  location of file
        :return: None
        """
        for key in self.list_clustered_bows.keys():
            if not os.path.exists(location):
                os.makedirs(location)
            location_np = os.path.join(location, key + '.txt')
            np.savetxt(location_np, self.list_clustered_bows[key])

    def populate_from_npy_files(self, location, port):
        """
        Populate BOW from file
        :param location: location of file
        :param port: port in EECVF from which to populate
        :return:
        """
        for key in self.list_clustered_bows.keys():
            dst_file = os.path.join(location, port, key + '.npy')
            self.list_clustered_bows[key] = np.load(dst_file)

    def add_gps_class(self, id, gps_string):
        """
        Add gps tag to clusters
        :param id: object id of cluster
        :param gps_string: gps tag in string format
        :return:
        """
        tmp = gps_string[0]
        longitude, latitude = tmp.split(';')
        longitude = float(longitude)
        latitude = float(latitude)
        if id not in self.list_gps_location.keys():
            self.list_gps_location[id] = dict()
            self.list_gps_location[id]['longitude'] = longitude
            self.list_gps_location[id]['latitude'] = latitude

    def save_gps(self, location, port_name):
        """
        Save gps tag cluster into file
        :param location: location of file
        :param port_name: port name
        :return:
        """
        if not os.path.exists(location):
            os.makedirs(location)
        location = os.path.join(location, port_name + '_gps.json')
        with open(location, 'w') as fp:
            json.dump(self.list_gps_location, fp)
        # np.savetxt(location, self.list_gps_location)

    def load_gps(self, location, port):
        """
        Load gps tag files from file
        :param location: location of file
        :param port: port name
        :return:
        """
        location = os.path.join(location, port, self.name + '_gps.json')
        with open(location, 'r') as fp:
            self.list_gps_location = json.load(fp)


###########################################################################################################################################
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
bow_cluster_list = dict()


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
    PORT_NR_CLASSES = 3
    # noinspection PyPep8Naming
    PORT_IN_SAVE_NPY = 4
    # noinspection PyPep8Naming
    PORT_IN_SAVE_TXT = 5
    # noinspection PyPep8Naming
    PORT_IN_GPS = 6
    # noinspection PyPep8Naming
    PORT_OUT_BOW = 7
    # verify that the number of parameters are OK.
    if len(param_list) != 8:
        log_error_to_console("ZuBuD_BOW JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_gps = get_port_from_wave(name=param_list[PORT_IN_GPS], wave_offset=param_list[PORT_IN_WAVE])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                global cluster_bow

                port_out = get_port_from_wave(name=param_list[PORT_OUT_BOW])

                if port_out.name not in cluster_bow.keys():
                    cluster_bow[port_out.name] = ZuBuD_BOW(name=port_in.name, dict_size=param_list[PORT_IN_DESC_SIZE], number_classes=param_list[PORT_NR_CLASSES])

                name = global_var_handler.PICT_NAME[:10]

                global object_list_des

                if name not in object_list_des.keys():
                    object_list_des[name] = list()
                    if param_list[PORT_IN_GPS] is not None:
                        cluster_bow[port_out.name].add_gps_class(name, port_in_gps.arr)

                object_list_des[name].append(port_in.arr)

                file_to_save = os.path.join(config_main.APPL_SAVE_LOCATION, port_out.name)
                if not os.path.exists(file_to_save):
                    os.makedirs(file_to_save)

                location_np = os.path.join(file_to_save, global_var_handler.PICT_NAME[:-4] + '_des.txt')
                np.savetxt(location_np, port_in.arr)

                if global_var_handler.NR_PICTURES - 1 == global_var_handler.FRAME:

                    bow_cluster_list[port_out.name] = list()

                    for key in object_list_des.keys():
                        BOW = cv2.BOWKMeansTrainer(param_list[PORT_IN_DESC_SIZE])

                        for el in object_list_des[key]:
                            BOW.add(np.float32(el))

                        desC = BOW.cluster()
                        bow_cluster_list[port_out.name].append(desC)
                        # only in last wave
                        BOW.clear()

                    keys = list(object_list_des.keys())
                    for idx in range(len(bow_cluster_list[port_out.name])):
                        cluster_bow[port_out.name].list_clustered_bows[keys[idx]] = bow_cluster_list[port_out.name][idx]

                    file_to_save = os.path.join(config_main.APPL_SAVE_LOCATION, port_out.name)

                    if param_list[PORT_IN_SAVE_TXT]:
                        cluster_bow[port_out.name].save_to_files_txt(file_to_save)

                    if param_list[PORT_IN_SAVE_NPY]:
                        cluster_bow[port_out.name].save_to_files_np(file_to_save)

                    if param_list[PORT_IN_GPS] is not None:
                        cluster_bow[port_out.name].save_gps(file_to_save, port_out.name)

                    t = np.array(list(cluster_bow[port_out.name].list_clustered_bows.values()))
                    port_out.arr[:] = t
                    port_out.set_valid()

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
    PORT_IN_DES_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_KP_POS = 1
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 3
    # noinspection PyPep8Naming
    PORT_IN_SAVED_TXT = 4
    # noinspection PyPep8Naming
    PORT_IN_SAVED_NPY = 5
    # noinspection PyPep8Naming
    PORT_IN_FLANN_THR = 6
    # noinspection PyPep8Naming
    PORT_NR_CLASSES = 7
    # noinspection PyPep8Naming
    PORT_IN_LOCATION_BOW = 8
    # noinspection PyPep8Naming
    PORT_IN_GPS = 9
    # noinspection PyPep8Naming
    PORT_IN_PORT_BOW = 10
    # noinspection PyPep8Naming
    PORT_OUT = 11
    # noinspection PyPep8Naming
    PORT_IN_DIST_OK = 12
    # noinspection PyPep8Naming
    PORT_IN_THR_MATCHING = 13
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 14
    # noinspection PyPep8Naming
    PORT_LOCATION_CSV_LANDMARKS = 15
    # noinspection PyPep8Naming
    PORT_SEMSEG = 16
    # noinspection PyPep8Naming
    PORT_IF_TRACKING = 17
    # noinspection PyPep8Naming
    PORT_IF_BOX = 18
    # verify that the number of parameters are OK.
    if len(param_list) != 19:
        log_error_to_console("ZuBuD_BOW_FLANN_INQUIRY JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_DES_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_gps = get_port_from_wave(name=param_list[PORT_IN_GPS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_kp = get_port_from_wave(name=param_list[PORT_IN_KP_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_img = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_semseg = get_port_from_wave(name=param_list[PORT_SEMSEG], wave_offset=param_list[PORT_IN_WAVE])

        global cluster_bow

        port_out = get_port_from_wave(name=param_list[PORT_OUT])
        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        # CHANGE IF NAME OF BOW CHANGES
        size_dict = int(param_list[PORT_IN_PORT_BOW].split('_')[2])

        cluster_bow[param_list[PORT_IN_PORT_BOW]] = ZuBuD_BOW(name=param_list[PORT_IN_PORT_BOW], dict_size=size_dict, is_from_file=True, number_classes=param_list[PORT_NR_CLASSES])

        if param_list[PORT_IN_SAVED_NPY]:
            cluster_bow[param_list[PORT_IN_PORT_BOW]].populate_from_npy_files(location=param_list[PORT_IN_LOCATION_BOW], port=param_list[PORT_IN_PORT_BOW])

        if param_list[PORT_IN_GPS] is not None:
            cluster_bow[param_list[PORT_IN_PORT_BOW]].load_gps(location=param_list[PORT_IN_LOCATION_BOW], port=param_list[PORT_IN_PORT_BOW])

        percent_class_dict = dict()
        matched_des = dict()

        # Blue color in BGR
        color_box = (255, 255, 0)
        color_text = (0, 255, 0)
        color_features = (0, 0, 255)

        for key in cluster_bow[param_list[PORT_IN_PORT_BOW]].list_clustered_bows.keys():
            try:
                # if True:
                if param_list[PORT_IN_GPS] is not None:
                    tmp = port_in_gps.arr[0]
                    longitude, latitude = tmp.split(';')
                    longitude = float(longitude)
                    latitude = float(latitude)
                    actual_location = (longitude, latitude)
                    landmark_location = (cluster_bow[param_list[PORT_IN_PORT_BOW]].list_gps_location[key]['longitude'],
                                         cluster_bow[param_list[PORT_IN_PORT_BOW]].list_gps_location[key]['latitude'])

                    distance = geopy.distance.geodesic(actual_location, landmark_location).m

                    if distance > param_list[PORT_IN_DIST_OK]:
                        percent_class_dict[key] = 0
                    else:
                        matched_des[key], percent_class_dict[key] = flann_matching(des1=port_in.arr,
                                                                                   des2=cluster_bow[param_list[PORT_IN_PORT_BOW]].list_clustered_bows[key],
                                                                                   comp_thr=param_list[PORT_IN_FLANN_THR])
                else:
                    matched_des[key], percent_class_dict[key] = flann_matching(des1=port_in.arr,
                                                                               des2=cluster_bow[param_list[PORT_IN_PORT_BOW]].list_clustered_bows[key],
                                                                               comp_thr=param_list[PORT_IN_FLANN_THR])
            except:
                percent_class_dict[key] = 0

        percent_class_dict = dict(sorted(percent_class_dict.items(), key=lambda item: item[1], reverse=True))

        t = np.array([[int(a[-4:]), float(x)] for a, x in percent_class_dict.items()])
        port_out.arr[:] = t

        list_images = list()

        if param_list[PORT_IN_KP_POS] is not None:
            tmp = list(percent_class_dict)
            new_kp = list()
            if percent_class_dict[tmp[0]] > 0:
                for m in matched_des[tmp[0]]:
                    new_kp.append(return_kp_cv2_object(port_in_kp.arr[m.queryIdx]))

                img1_miss_matched_kp = cv2.drawKeypoints(port_in_img.arr, new_kp, None, color=color_features, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                file = open(param_list[PORT_LOCATION_CSV_LANDMARKS], "r")
                # get dictionary fields from csv header
                fields = file.readline().split(',')
                # eliminate new line character
                fields[-1] = fields[-1].split('\n')[0]

                for line in file.readlines():
                    data = line.split(',')
                    new_obj = dict()

                    for idx_field in range(len(fields)):
                        new_obj[fields[idx_field]] = data[idx_field]

                    list_images.append(new_obj)

                file.close()

                if port_in_semseg is None:
                    port_out_img.arr[:] = img1_miss_matched_kp
                else:
                    new_semseg_img = np.zeros((port_in_semseg.arr.shape[0], port_in_semseg.arr.shape[1], 3), dtype=np.uint8)
                    new_semseg_img[:, :, 2] = (port_in_semseg.arr != 1) * 125
                    port_out_img.arr[:] = cv2.addWeighted(new_semseg_img, 0.3, img1_miss_matched_kp, 0.7, 0)

                if param_list[PORT_IF_BOX] and param_list[PORT_IF_TRACKING] == False:
                    if len(new_kp) != 0:
                        min_left = None
                        min_up = None
                        max_right = None
                        max_down = None
                        for kp in new_kp:
                            if min_left == None:
                                min_left = kp.pt[0]
                                max_right = kp.pt[0]
                                min_up = kp.pt[1]
                                max_down = kp.pt[1]
                            else:
                                if min_left > kp.pt[0]:
                                    min_left = kp.pt[0]
                                elif max_right < kp.pt[0]:
                                    max_right = kp.pt[0]

                                if min_up > kp.pt[1]:
                                    min_up = kp.pt[1]
                                elif max_down < kp.pt[1]:
                                    max_down = kp.pt[1]

                        start_point = (int(min_left * 0.75), int(min_up * 0.85))
                        end_point = (int(max_right * 1.25), int(max_down * 1.15))


                        # Line thickness of 2 px
                        thickness = 1

                        port_out_img.arr[:] = cv2.rectangle(img=port_out_img.arr, pt1=start_point, pt2=end_point, color=color_box, thickness=thickness)

                x = int(tmp[0][-4:])

                if param_list[PORT_IF_TRACKING]:
                    global frame_detection_tracker

                    if param_list[PORT_IN_PORT_BOW] not in frame_detection_tracker.keys():
                        frame_detection_tracker[param_list[PORT_IN_PORT_BOW]] = frame_tracker()

                    frame_detection_tracker[param_list[PORT_IN_PORT_BOW]].decay_frame_tracker()

                    if percent_class_dict[tmp[0]] >= param_list[PORT_IN_THR_MATCHING]:
                        frame_detection_tracker[param_list[PORT_IN_PORT_BOW]].add_to_frame_tracker(x)
                    else:
                        frame_detection_tracker[param_list[PORT_IN_PORT_BOW]].decay_all()

                    x, text_to_show = frame_detection_tracker[param_list[PORT_IN_PORT_BOW]].get_frame_max_value()
                else:
                    x, text_to_show = x, ""

                if x is not None:

                    if param_list[PORT_IF_BOX]:
                        if len(new_kp) != 0:
                            min_left = None
                            min_up = None
                            max_right = None
                            max_down = None

                            for kp in new_kp:
                                if min_left == None:
                                    min_left = kp.pt[0]
                                    max_right = kp.pt[0]
                                    min_up = kp.pt[1]
                                    max_down = kp.pt[1]
                                else:
                                    if min_left > kp.pt[0]:
                                        min_left = kp.pt[0]
                                    elif max_right < kp.pt[0]:
                                        max_right = kp.pt[0]

                                    if min_up > kp.pt[1]:
                                        min_up = kp.pt[1]
                                    elif max_down < kp.pt[1]:
                                        max_down = kp.pt[1]

                            start_point = (max(int(min_left * 0.75), 2), max(int(min_up * 0.85),2))
                            end_point = (min(int(max_right * 1.25), port_out_img.arr.shape[1] - 2) , min(int(max_down * 1.15), port_out_img.arr.shape[0]-2))

                            # Line thickness of 2 px
                            thickness = 1

                            port_out_img.arr[:] = cv2.rectangle(img=port_out_img.arr, pt1=start_point, pt2=end_point, color=color_box, thickness=thickness)

                    try:
                        text = list_images[x - 1]['Building Name'][:-1]
                    except:
                        print('!!!!!!!!!!!!!!!!!!!!!!!', x)
                        text = ''
                    fontScale = min(port_out_img.arr.shape[0], port_out_img.arr.shape[1]) / 900
                    port_out_img.arr[:] = cv2.putText(port_out_img.arr, text_to_show, (int(port_out_img.arr.shape[1] * 0.005), int(port_out_img.arr.shape[0] * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_text, 1, cv2.LINE_AA)

                else:
                    text = 'None'
            else:
                port_out_img.arr = cv2.drawKeypoints(port_in_img.arr, [], None, color=color_features, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                text = 'None'

            fontScale = min(port_out_img.arr.shape[0], port_out_img.arr.shape[1]) / 800
            port_out_img.arr[:] = cv2.putText(port_out_img.arr, text, (int(port_out_img.arr.shape[1] * 0.005), int(port_out_img.arr.shape[0] * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_text, 1, cv2.LINE_AA)

            port_out_img.set_valid()

            # cv2.imshow('image',port_out_img.arr)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        file_to_save = os.path.join(config_main.APPL_SAVE_LOCATION, port_out.name)
        if not os.path.exists(file_to_save):
            os.makedirs(file_to_save)
        location_np = os.path.join(file_to_save, global_var_handler.PICT_NAME.split('.')[0] + '.txt')
        np.savetxt(fname=location_np, X=port_out.arr, fmt=['%3.0f', '%3.3f'])

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

def do_tmbud_bow_job(port_to_add: str, dictionary_size: int, number_classes: int = 201,
                     use_gps=False, gps_port=None,
                     save_to_text: bool = True, save_to_npy: bool = True,
                     port_bow_list_output: str = None,
                     level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Bag  of  Features  (BOF)  has  become  a  popular  approach in  CV  for  image  classification,  object  detection,  or  image retrieval.
    The  name  comes  from  the  Bag  of  Words  (BOW)representation used in textual information retrieval
    # https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/csurka-eccv-04.pdf
    :param port_to_add:  Feature port to add to BOW
    :param dictionary_size: size of cluster per class
    :param number_classes: number of cluster elements
    :param use_gps: if gps data exist
    :param gps_port: Name of gps port
    :param save_to_text: if we want to save BOW clusters to txt files
    :param save_to_npy: if we want to save BOW clusters to npy files
    :param port_bow_list_output: output port with BOW
    :param level: pyramid level to calculate at
    :param wave_offset: wave of input port, please correlate with each input port name parameter
    :return: Name of output port or ports
    """
    # Do this for each input port this function has
    input_port_name = transform_port_name_lvl(name=port_to_add, lvl=level)
    input_port_list = [input_port_name]

    if use_gps is True:
        gps_port_name = transform_port_name_lvl(name=gps_port, lvl=level)
        input_port_list.append(gps_port_name)
    else:
        gps_port_name = None

    if port_bow_list_output is None:
        port_output = '{name}_{size}_{Input}'.format(name='ZuBuD_BOW', size=dictionary_size.__str__(), Input=port_to_add)

    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_output, lvl=level)

    port_des_output_name_size = '({nr_classes}, {size}, 64)'.format(nr_classes=number_classes, size=dictionary_size)
    output_port_list = [(port_img_output_name, port_des_output_name_size, 'f', False)]
    main_func_list = [input_port_name, wave_offset, dictionary_size, number_classes, save_to_text, save_to_npy, gps_port_name, port_img_output_name]

    job_name = job_name_create(action='ZuBuD_BOW {size}'.format(size=dictionary_size.__str__()),
                               input_list=input_port_list[:1], wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='bow_zubud_main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output


def do_tmbud_bow_inquiry_flann_job(port_to_inquiry_des: str,
                                   flann_thr: float, location_of_bow: str, bow_port: str,
                                   port_to_inquiry_kp: str = None, port_to_inquire_img: str = None, save_img_detection: bool = False,
                                   saved_to_text: bool = False, saved_to_npy: bool = True, number_classes: int = 201,
                                   use_gps: bool = False, gps_port: str = None, distante_accepted: int = 10,
                                   if_tracking: bool = False, if_box_on_detection: bool = False,
                                   threshold_matching: int = 5, name_landmark_port: str = None, mask_port: str = None,
                                   port_out_name: str = None, port_out_image: str = None,
                                   level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Bag  of  Features  (BOF)  has  become  a  popular  approach in  CV  for  image  classification,  object  detection,  or  image retrieval.
    The  name  comes  from  the  Bag  of  Words  (BOW)representation used in textual information retrieval
    # https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/csurka-eccv-04.pdf
    :param port_to_inquiry_des:  descriptor list for port we wish to inquire
    :param port_to_inquiry_kp:  key-point list for port we wish to inquire
    :param port_to_inquire_img:  image port we wish to inquire
    :param save_img_detection: if we want to save image with details of clasification
    :param use_gps: if gps data exist
    :param gps_port: Name of gps port
    :param distante_accepted: distance in meters accepted
    :param flann_thr: Threshold for FLANN matching
    :param saved_to_text: if the BOW clusters were saved in txt files
    :param saved_to_npy: if the BOW clusters were saved in npy files
    :param location_of_bow: location of BOW saved files
    :param bow_port: port of BOW
    :param port_out_name: output port with BOW
    :param port_out_image: output image port with BOW
    :param number_classes: number of cluster of BOW
    :param threshold_matching: value used to filter out detection based on percent
    :param name_landmark_port: port were one can find the landmark names
    :param mask_port: semseg port
    :param if_tracking: if we want tracking to be active
    :param if_box_on_detection: if we want to put a box in the image on the detection
    :param level: pyramid level to calculate at
    :param wave_offset: wave of input port, please correlate with each input port name parameter
    :return: Name of output port or ports
    """
    # Do this for each input port this function has
    input_des_port_name = transform_port_name_lvl(name=port_to_inquiry_des, lvl=level)
    input_port_list = [input_des_port_name]

    if use_gps is True:
        gps_port_name = transform_port_name_lvl(name=gps_port, lvl=level)
        input_port_list.append(gps_port_name)
    else:
        gps_port_name = None

    if save_img_detection is True:
        input_kp_port_name = transform_port_name_lvl(name=port_to_inquiry_kp, lvl=level)
        input_img_port_name = transform_port_name_lvl(name=port_to_inquire_img, lvl=level)
        input_port_list.append(input_kp_port_name)
        input_port_list.append(input_img_port_name)

        if mask_port is not None:
            input_semseg_port_name = transform_port_name_lvl(name=mask_port, lvl=level)
            input_port_list.append(input_semseg_port_name)
        else:
            input_semseg_port_name = None
    else:
        input_kp_port_name = None
        input_img_port_name = None

    if port_out_name is None:
        port_out_name = 'ZuBuD_BOW_INQ_THR_{thr}_DIST_{dist}_{Input}'.format(thr=flann_thr.__str__().replace('.', '_'), Input=bow_port[0:-3], dist=int(distante_accepted).__str__())

    port_output_name = transform_port_name_lvl(name=port_out_name, lvl=level)
    port_des_output_name_size = '({nr_classes}, 2)'.format(nr_classes=number_classes)

    if save_img_detection is True:
        if port_out_image is None:
            port_out_image = 'ZuBuD_BOW_INQ_IMG_THR_{thr}_DIST_{dist}_{Input}'.format(thr=flann_thr.__str__().replace('.', '_'), Input=bow_port[0:-3], dist=distante_accepted.__str__())

        port_out_image_name = transform_port_name_lvl(name=port_out_image, lvl=level)
        port_out_image_size = transform_port_size_lvl(lvl=level, rgb=True)
    else:
        port_out_image_name = None

    main_func_list = [input_des_port_name, input_kp_port_name, input_img_port_name, wave_offset, saved_to_text, saved_to_npy, flann_thr, number_classes,
                      location_of_bow, gps_port_name, bow_port, port_output_name, distante_accepted, threshold_matching, port_out_image_name, name_landmark_port, input_semseg_port_name, if_tracking, if_box_on_detection]
    output_port_list = [(port_output_name, port_des_output_name_size, 'f', False)]

    if save_img_detection is True:
        output_port_list.append((port_out_image_name, port_out_image_size, 'B', True))

    job_name = job_name_create(action='ZuBuD_BOW inquiry FLANN {thr}'.format(thr=flann_thr.__str__().replace('.', '_')),
                               input_list=input_port_list[:1], wave_offset=[wave_offset], level=level)

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
