import cv2
import os
import config_main as CONFIG


def create_video(port_list, folder_of_ports, name, fps, concat_vertical=False, concat_horizontal=True):
    """
    Creates video out of ports
    :param port_list: list of ports to transform
    :param folder_of_ports: location of saved images
    :param name: output name of video
    :param fps: desired fps speed
    :param concat_vertical: if there are multiple ports and we want to concat vertical
    :param concat_horizontal: if there are multiple ports and we want to concat horizontal
    :return:
    """
    list_of_folder = list()
    height_list = list()
    width_list = list()
    new_width = 0
    new_height = 0

    for port in port_list:
        path = os.path.join(folder_of_ports, port)
        list_of_folder.append([os.path.join(path,img) for img in os.listdir(path) if img.endswith(".png")])

    # print(list_of_folder)

    for idx in range(len(port_list)):
        frame = cv2.imread(list_of_folder[idx][0])
        height, width, layers = frame.shape
        height_list.append(height)
        width_list.append(width)

    # print('height_list', height_list)
    # print('width_list', width_list)

    if concat_horizontal:
        new_width = sum(width_list)
        new_height = max(height_list)
    elif concat_vertical:
        new_width = max(width_list)
        new_height = sum(height_list)
    else:
        new_width = max(width_list)
        new_height = max(height_list)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter('c:/repos/eecvf_git/Logs/{}'.format(name), fourcc, frameSize=(new_width, new_height), fps=fps)

    for idx in range(len(list_of_folder[0])):
        curent_frame_list = list()
        for idxx in range(len(list_of_folder)):
            curent_frame_list.append(cv2.imread(os.path.join(list_of_folder[idxx][idx])))
        if concat_horizontal:
            new_img = cv2.hconcat(curent_frame_list)
        elif concat_vertical:
            new_img = cv2.vconcat(curent_frame_list)
        else:
            new_img = curent_frame_list

        # cv2.imshow('ex', new_img)
        # cv2.waitKey(0)

        video.write(new_img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    # create_video(port_list=['RAW_L2', 'OVERLAY_SEMSEG_RAW_L2', 'FINAL_L2'],
    #              folder_of_ports='C:/repos/eecvf_git/Logs/query_application',
    #              name='1.mp4',
    #              fps=25)

    create_video(port_list=['FINAL_L2'],
                 folder_of_ports='C:/repos/eecvf_git/Logs/query_application',
                 name='2.mp4',
                 fps=25)