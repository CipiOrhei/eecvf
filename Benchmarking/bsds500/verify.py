"""
Verify the functionality of the evaluation suite.

Executes the evaluation procedure against five samples and outputs the
results. Compare them with the results from the BSDS dataset to verify
that this Python port works properly.
"""
import os
import config_main

import tqdm
from Benchmarking.bsds500.bsds.bsds_dataset import BSDSDataset
from Benchmarking.bsds500.bsds import evaluate_boundaries
from skimage.util import img_as_float
from skimage.io import imread
from Utils.log_handler import log_error_to_console, log_benchmark_info_to_console, log_setup_info_to_console


def load_gt_boundaries(sample_name):
    gt_path = os.path.join(os.getcwd(), config_main.BENCHMARK_GT_LOCATION, '{}.mat'.format(sample_name))
    return BSDSDataset.load_boundaries(gt_path)


def load_pred(sample_name, set_name):
    pred_path = os.path.join(config_main.BENCHMARK_INPUT_LOCATION, set_name, '{}.png'.format(sample_name))
    return img_as_float(imread(pred_path))


def run_verify_boundry(thinning, max_distance_px):
    log_setup_info_to_console("BENCHMARKING BSDS500 BOUNDRY")

    idx = 1
    for set in config_main.BENCHMARK_SETS:
        log_benchmark_info_to_console('Current set: {id}/{max}: {set}'.format(id=idx, max=len(config_main.BENCHMARK_SETS),
                                                                            set=set))
        try:
            sample_results, threshold_results, overall_result = \
                evaluate_boundaries.pr_evaluation(thresholds=config_main.BENCHMARK_BSDS_500_N_THRESHOLDS,
                                                  sample_names=config_main.BENCHMARK_SAMPLE_NAMES,
                                                  set_name=set,
                                                  load_gt_boundaries=load_gt_boundaries,
                                                  load_pred=load_pred,
                                                  thinning=thinning,
                                                  max_distance_px=max_distance_px,
                                                  progress=tqdm.tqdm)
            idx += 1

            # Write results to disk
            results_path = os.path.join(os.getcwd(), config_main.BENCHMARK_RESULTS, 'PCM')

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            file = open(os.path.join(results_path, set + '.log'), "w+")

            file.write('Per image (#, threshold, recall, precision, f1):\n\n')

            for sample_index, res in enumerate(sample_results):
                file.write('{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}\n'.format(sample_index + 1, res.threshold,
                                                                                      res.recall, res.precision, res.f1))
            file.write('\nOverall (threshold, recall, precision, f1):\n\n')

            for thresh_i, res in enumerate(threshold_results):
                file.write('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}\n'.format(res.threshold, res.recall,
                                                                              res.precision, res.f1))

            file.write('\nSummary (overall threshold, overall recall, overall precision, overall f1, overall best recall, '
                       'overall best precision, overall best f1, area_pr):\n')
            log_benchmark_info_to_console('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
                overall_result.threshold, overall_result.recall, overall_result.precision, overall_result.f1,
                overall_result.best_recall, overall_result.best_precision, overall_result.best_f1,
                overall_result.area_pr)
            )
            file.write('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}\n'.format(
                overall_result.threshold, overall_result.recall, overall_result.precision, overall_result.f1,
                overall_result.best_recall, overall_result.best_precision, overall_result.best_f1,
                overall_result.area_pr)
            )
        except Exception as ex:
            log_error_to_console('BENCHMARK NOK', ex.__str__())


if __name__ == "__main__":
    pass
