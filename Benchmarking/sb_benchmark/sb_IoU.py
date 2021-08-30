import os

# noinspection PyPackageRequirements
import cv2
import numpy as np
import json

import config_main
from Utils.log_handler import log_setup_info_to_console, log_error_to_console, log_benchmark_info_to_console
from Benchmarking.Util.image_parsing import find_img_extension


def sb_iou(box1, box2) -> float:
    x1, y1, x2, y2 = box1[0][0], box1[0][1], box1[1][0], box1[1][1]
    x3, y3, x4, y4 = box2[0][0], box2[0][1], box2[1][0], box2[1][1]
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2- y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou


# noinspection PyPep8Naming
def run_SB_benchmark_IoU() -> None:
    """
    Calculate the IoU for a set of images.
    :param class_list_name: list of names corresponding the classes
    :param unknown_class: the class value for unknown classification of pixels.
    :param is_rgb_gt: if ground truth images are in RGB format
    :param class_list_rgb_value: list of greyscale values of RGB colors used.
    :param show_only_set_mean_value: show on console only the mean IoU for each set
    :return: None
    """
    log_setup_info_to_console("BENCHMARKING SB IoU")

    ground_truth = dict()
    json_files = [f for f in os.listdir(config_main.BENCHMARK_GT_LOCATION) if os.path.isfile(os.path.join(config_main.BENCHMARK_GT_LOCATION, f))]

    try:
        for json_file in json_files:
            f = open(os.path.join(config_main.BENCHMARK_GT_LOCATION, json_file))
            data = json.load(f)
            f.close()
            ground_truth[(data["asset"]["name"]).split('.')[0]] = data["regions"][0]["points"]
    except KeyError as e:
        log_error_to_console('BENCHMARK SB IoU NOK: Key Not Found', e.__str__())

    for set_image in config_main.BENCHMARK_SETS:
        # log_benchmark_info_to_console('Current set: {}'.format(set_image))

        # try:
        if True:
            # Write results to disk
            results_path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, 'SB_IoU')

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            out = open(os.path.join(results_path, set_image + '.log'), "w+")
            out.write('Per image (#, IoU):\n')

            iou_mean = 0
            iou_nr = 0
            tp = 0
            tn = 0
            fp = 0
            fp_m = 0
            fn = 0
            tp_m = 0

            for file in config_main.BENCHMARK_SAMPLE_NAMES:
                # TODO Adapt to name changes of json
                gt_boxes = list()
                algo_boxes = list()
                try:
                    # print("DEBUG DATA: FILE ", file)
                    # get data from json of predicted boxes
                    path = os.path.join(config_main.APPL_SAVE_LOCATION + '/' + set_image, ''.join(file + ".json"))
                    f = open(path, encoding='utf8')
                    data = json.load(f)
                    f.close()
                    # add gt_boxes

                    if file in ground_truth.keys():
                        gt_boxes.append([[ground_truth[file][3]['x'], ground_truth[file][3]['y']], [ground_truth[file][1]['x'], ground_truth[file][1]['y']]])
                    else:
                        gt_boxes = None
                    # add algo_boxes
                    for box in range(len(data['regions'])):
                        algo_boxes.append([[data["regions"][box]["points"][0]['x'], data["regions"][box]["points"][0]['y']], [data["regions"][box]["points"][2]['x'], data["regions"][box]["points"][2]['y']]])
                    if len(algo_boxes) == 0:
                        algo_boxes = None
                    # print("DEBUG DATA: gt_boxes ", gt_boxes)
                    # print("DEBUG DATA: algo_boxes ", algo_boxes)
                except Exception as e:
                    gt_boxes = None
                    algo_boxes = None
                    log_error_to_console('BENCHMARK SB IoU NOK', e.__str__())
                    pass
                # this works on the presumption that we have only one gt box
                tmp_iou = [0.000]
                if gt_boxes == None and algo_boxes == None:
                    tmp_iou = [1.00]
                    tn += 1
                elif gt_boxes == None and algo_boxes != None:
                    tmp_iou = [0.00]
                    fp += 1
                    fp_m += 1
                elif gt_boxes != None and algo_boxes == None:
                    tmp_iou = [0.00]
                    fn += 1
                else:
                    for i in range(len(algo_boxes)):
                        tmp = 0.0
                        try:
                            tmp = sb_iou(box1=algo_boxes[i], box2=gt_boxes[0])
                        except Exception as ex:
                            log_error_to_console('BENCHMARK SB IoU NOK', ex.__str__())
                            pass
                        tmp_iou.append(tmp)
                    if len(algo_boxes) != len(gt_boxes):
                        fp += 1
                    else:
                        tp += 1

                    tp_m += 1

                iou = max(tmp_iou)
                # log_benchmark_info_to_console('IoU: {:<20s} \t {:s}\n'.format(file, str(iou)))
                out.write('IoU: {:<20s} \t {:s}\n'.format(file, str(iou)))
                iou_mean += iou
                iou_nr += 1

            avg_iou = iou_mean / iou_nr
            out.write("Mean: " + str(avg_iou))
            # out.close()
            log_benchmark_info_to_console('IoU: {:<20s} \t {:s}\n'.format(set_image, str(avg_iou)))

            acc = (tp + tn) / (tp + tn + fp + fn)
            p = tp/(tp + fp)
            r = tp/(tp + fn)
            f1 = 2*p*r/(p+r)
            out.write("Acc: " + str(acc))
            log_benchmark_info_to_console('Acc: {:<20s} \t {:s}\n'.format(set_image, str(acc)))
            log_benchmark_info_to_console('P: {:<20s} \t {:s}\n'.format(set_image, str(p)))
            log_benchmark_info_to_console('R: {:<20s} \t {:s}\n'.format(set_image, str(r)))
            log_benchmark_info_to_console('F1: {:<20s} \t {:s}\n'.format(set_image, str(f1)))

            acc_m = (tp_m + tn) / (tp_m + tn + fp_m + fn)
            out.write("Acc_m: " + str(acc_m))
            log_benchmark_info_to_console('Acc_m: {:<20s} \t {:s}\n'.format(set_image, str(acc_m)))

            out.close()
        # except Exception as ex:
        #     log_error_to_console('BENCHMARK SB IoU NOK', ex.__str__())


if __name__ == "__main__":
    pass