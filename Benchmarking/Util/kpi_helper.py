

def calc_tpr(tp: float, fn: float) -> float:
    """
    :param tp: true positive or hit
    :param fn: false negative miss
    :return: sensitivity or true positive rate
    """
    try:
        calc = tp / (tp + fn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_recall(tp: float, fn: float) -> float:
    """
    :param tp: true positive or hit
    :param fn: false negative miss
    :return: recall or true positive rate
    """
    try:
        calc = calc_tpr(tp, fn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_sensitivity(tp: float, fn: float) -> float:
    """
    :param tp: true positive or hit
    :param fn: false negative miss
    :return: sensitivity or true positive rate
    """
    try:
        calc = calc_tpr(tp, fn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_hit_rate(tp: float, fn: float) -> float:
    """
    :param tp: true positive or hit
    :param fn: false negative miss
    :return: hit rate or true positive rate
    """
    try:
        calc = calc_tpr(tp, fn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_tnr(tn: float, fp: float) -> float:
    """
    :param tn: true negative or correct rejection
    :param fp: false positive or false alarm
    :return: true negative rate
    """
    try:
        calc = tn / (tn + fp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_specificity(tn: float, fp: float) -> float:
    """
    :param tn: true negative or correct rejection
    :param fp: false positive or false alarm
    :return: specificity or true negative rate
    """
    try:
        calc = calc_tnr(tn, fp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_selectivity(tn: float, fp: float) -> float:
    """
    :param tn: true negative or correct rejection
    :param fp: false positive or false alarm
    :return: selectivity or true negative rate
    """
    try:
        calc = calc_tnr(tn, fp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_precision(tp: float, fp: float) -> float:
    """
    :param tp: true positive or hit
    :param fp: false positive or false alarm
    :return: precision
    """
    try:
        calc = tp / (tp + fp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_ppv(tp: float, fp: float) -> float:
    """
    :param tp: true positive or hit
    :param fp: false positive or false alarm
    :return: precision
    """
    try:
        calc = calc_precision(tp, fp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_npv(tn: float, fn: float) -> float:
    """
    :param tn: true negative or correct rejection
    :param fn: false negative miss

    :return: negative predictive value
    """
    try:
        calc = tn / (tn + fn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_fnr(fn: float, tp: float) -> float:
    """
    :param fn: false negative miss
    :param tp: true positive or hit

    :return: false negative rate
    """
    try:
        calc = fn / (fn + tp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_miss_rate(fn: float, tp: float) -> float:
    """
    :param fn: false negative miss
    :param tp: true positive or hit

    :return: miss rate or false negative rate
    """
    try:
        calc = calc_fnr(fn, tp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_fpr(fp: float, tn: float) -> float:
    """
    :param fp: false positive or false alarm
    :param tn: true negative or correct rejection

    :return: false positive rate
    """
    try:
        calc = fp / (fp + tn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_fall_out(fp: float, tn: float) -> float:
    """
    :param fp: false positive or false alarm
    :param tn: true negative or correct rejection

    :return: fall-out or false positive rate
    """
    try:
        calc = calc_fpr(fp, tn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_fdr(fp: float, tp: float) -> float:
    """
    :param fp: false positive or false alarm
    :param tp: true positive or hit

    :return: false discovery rate
    """
    try:
        calc = fp / (fp + tp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_for(fn: float, tn: float) -> float:
    """
    :param fn: false negative miss
    :param tn: true negative or correct rejection

    :return: false omission rate
    """
    try:
        calc = fn / (fn + tn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_ts(fn: float, fp: float, tp: float) -> float:
    """
    :param fn: false negative miss
    :param fp: false positive or false alarm
    :param tp: true positive or hit

    :return: Threat score
    """
    try:
        calc = tp / (tp + fn + fp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_csi(fn: float, fp: float, tp: float) -> float:
    """
    :param fn: false negative miss
    :param fp: false positive or false alarm
    :param tp: true positive or hit

    :return: Critical Success Index or Threat score
    """
    try:
        calc = tp / (tp + fn + fp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_acc(fn: float, fp: float, tp: float, tn: float) -> float:
    """
    :param fn: false negative miss
    :param fp: false positive or false alarm
    :param tp: true positive or hit
    :param tn: true negative or correct rejection

    :return: accuracy
    """
    try:
        calc = (tp + tn) / (tp + tn + fn + fp)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_f1_score(fn: float, fp: float, tp: float) -> float:
    """
    :param fn: false negative miss
    :param fp: false positive or false alarm
    :param tp: true positive or hit

    :return: is the harmonic mean of precision and sensitivity
    """
    try:
        calc = (2 * tp) / (2 * tp + fp + fn)
    except ZeroDivisionError:
        calc = 0

    return calc


def calc_f_measure(recall, precission):
    """
    :param recall:
    :param precission:
    :return: compute f-measure fromm recall and precision
    """
    try:
        calc = (2 * precission * recall) / (precission + recall + (precission + recall == 0) * 1.0)
    except ZeroDivisionError:
        calc = 0

    return calc


if __name__ == "__main__":
    pass
