import os
# noinspection PyPackageRequirements
import cv2


def calc_acc(fn: float, fp: float, tp: float, tn: float) -> float:
    """
    :param fn: false negative miss
    :param fp: false positive or false alarm
    :param tp: true positive or hit
    :param tn: true negative or correct rejection

    :return: accuracy
    """
    return (tp + tn) / (tp + tn + fn + fp)


def calc_precision(tp: float, fp: float) -> float:
    """
    :param tp: true positive or hit
    :param fp: false positive or false alarm
    :return: precision
    """
    if tp + fp != 0:
        return tp / (tp + fp)
    else:
        return 0


def calc_tpr(tp: float, fn: float) -> float:
    """
    :param tp: true positive or hit
    :param fn: false negative miss
    :return: sensitivity or true positive rate
    """
    return tp / (tp + fn)


def calc_f1_score(fn: float, fp: float, tp: float) -> float:
    """
    :param fn: false negative miss
    :param fp: false positive or false alarm
    :param tp: true positive or hit

    :return: is the harmonic mean of precision and sensitivity
    """
    return (2 * tp) / (2 * tp + fp + fn)


def get_files_in_folder(folder):
    """
    Updates the global variable images_in_directory with all the frames inside.
    :param folder: source directory for the input pictures
    :return: None
    """
    list_of_files = []
    for dir_name, dir_names, file_names in os.walk(folder):
        # array that stores all names
        for filename in file_names:
            list_of_files.append(filename)

    return list_of_files


def get_folders_in_folder(folder):
    """
    Updates the global variable images_in_directory with all the frames inside.
    :param folder: source directory for the input pictures
    :return: None
    """
    list_of_folders = []
    for dir_name, dir_names, file_names in os.walk(folder):
        # array that stores all names
        for dir in dir_names:
            list_of_folders.append(dir)

    return list_of_folders


def matrix_differences(img1, img2):
    img = img1 - img2
    # 2 pt ca 0-255 = 1
    ret, new_img = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY)

    return new_img


def matrix_reunion(img1, img2):
    img = img1 + img2

    return img


def calculate_kpi(image, gt):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gt = cv2.imread(gt)
    if len(img_gt.shape) == 3:
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)

    size = len(img) * len(img[0])

    fp_map = matrix_differences(img, img_gt)
    fn_map = matrix_differences(img_gt, img)
    tp_map = matrix_differences(img, fp_map)
    tn_map = matrix_reunion(img, img_gt)

    fp = cv2.countNonZero(fp_map)
    fn = cv2.countNonZero(fn_map)
    tp = cv2.countNonZero(tp_map)
    tn = (size - cv2.countNonZero(tn_map))

    # fp = 0
    # fn = 0
    # tp = 0
    # tn = 0
    #
    # img = img != 0
    # img_gt = img_gt != 0
    #
    # tp = img == True and img_gt == True
    # tn = img == False and img_gt == False
    # fp = img == True and img_gt == False
    # fn = img == False and img_gt == True
    #
    # print(tp)

    # for row in range(img.shape[0]):
    #     for column in range(img.shape[1]):
    #         if img[row][column] != 0 and img_gt[row][column] != 0:
    #             tp += 1
    #         elif img[row][column] == 0 and img_gt[row][column] == 0:
    #             tn += 1
    #         elif img[row][column] != 0 and img_gt[row][column] == 0:
    #             fp += 1
    #         elif img[row][column] == 0 and img_gt[row][column] != 0:
    #             fn += 1

    return tp, tn, fp, fn, size


def calculate_kpi_set(input_folder, gt_folder):
    input_images = get_files_in_folder(input_folder)

    c = ''
    for i in input_images:
        if i.endswith('12.png'):
            c = i

    input_images.remove(c)

    tp_global = 0
    tn_global = 0
    fp_global = 0
    fn_global = 0
    size_global = 0

    for file in input_images:
        tp, tn, fp, fn, size = calculate_kpi(os.path.join(input_folder, file), os.path.join(gt_folder, file.split('_')[-1]))

        tp_global += tp
        tn_global += tn
        fp_global += fp
        fn_global += fn
        size_global += size
        # break

    # tp_global /= size_global
    # tn_global /= size_global
    # fp_global /= size_global
    # fn_global /= size_global
    tpr = calc_tpr(tp=tp_global, fn=fn_global)
    acc = calc_acc(fn_global, fp_global, tp_global, tn_global)
    pre = calc_precision(tp_global, fp_global)
    f1 = calc_f1_score(fn=fn_global, fp=fp_global, tp=tp_global)

    text = '{set}, TPR, {tpr}, P, {p}, Acc, {ac}, F1, {f}'.format(
        set=input_folder.split('/')[-1], tpr=tpr, ac=acc, p=pre, f=f1)
    print(text)
    return text


if __name__ == "__main__":
    pass
