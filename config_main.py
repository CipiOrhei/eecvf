"""
Module handles the configuration of the EECVF.
Please do not change this file, only if necessary and service does not exist for that change.
"""

CUDA_GPU = True

JSON_FILE_LOCATION = 'Application//Config//json'
JSON_FILE_NAME = 'new_config'

# Application configuration
IMAGE_INPUT = 0
VIDEO_INPUT = 1
CAMERA_INPUT = 2
IMAGE_TXT_INPUT = 3
# Please take care that this values are rewritten by the service jobs
# Input l
APPL_INPUT_DIR = 'TestData/smoke_test'
# if input image/video/camera
APPL_INPUT = None
# video as input
APPL_INPUT_VIDEO = ''
# number of frames to run from video camera
APPL_NR_FRAMES_CAPTURE = 0
# folder for image input
APPL_INPUT_IMG_DIR = []
# job json file relative to Application
APPL_INPUT_JOB_LIST = ''
# Number of waves to support
APPL_NR_WAVES = 1
# pictures to show
APPL_SHOW_PICT = False
APPL_SHOW_LIST = []
APPL_SHOW_TIME = 0  # 0 to hold picture.
# pictures to save
APPL_SAVE_PICT = True
APPL_SAVE_PICT_LIST = []
APPL_SAVE_JOB_NAME = False
APPl_SAVE_PICT_EXTENSION = '.png'
APPL_SAVE_LOCATION = 'Logs/application_results'  # use '' for default

APPL_SATELLITE_IMAGE_PROCESSING = False

ML_TRAIN_IMG_LOCATION = ''
ML_TEST_IMG_LOCATION = ''
ML_VALIDATE_IMG_LOCATION = ''
ML_LABEL_IMG_LOCATION = ''
ML_OUTPUT_IMG_LOCATION = 'Logs/ml_results'
ML_WEIGHT_OUTPUT_LOCATION = 'MachineLearning/model_weights'

# benchmark configuration
BENCHMARK_GT_LOCATION = ''
BENCHMARK_INPUT_LOCATION = APPL_SAVE_LOCATION
BENCHMARK_RESULTS = 'Logs/benchmark_results'

BENCHMARK_SAMPLE_NAMES = []
BENCHMARK_SETS = []
BENCHMARK_BSDS_500_N_THRESHOLDS = 5

# log files configuration
LOG_FILE = 'Logs/console.log'
LOG_KPI_FILE = 'Logs/log.csv'


class PYRAMID_LEVEL:
    LEVEL_0 = 'L0'
    LEVEL_1 = 'L1'
    LEVEL_2 = 'L2'
    LEVEL_3 = 'L3'
    LEVEL_4 = 'L4'
    LEVEL_5 = 'L5'
    LEVEL_6 = 'L6'
    LEVEL_7 = 'L7'
    LEVEL_8 = 'L8'

    NUMBER_LVL = 8


class MORPH_CONFIG:
    # configuration to choose for kernels type
    KERNEL_RECTANGULAR = 'cv2.MORPH_RECT'
    KERNEL_CROSS = 'cv2.MORPH_CROSS'
    KERNEL_ELLIPSE = 'cv2.MORPH_ELLIPSE'
    # configuration to choose for kernels for hit-miss
    KERNEL_HIT_MISS = [[0, 1, 0], [-1, 1, 1], [-1, -1, 0]]
    # configuration to choose for thinning
    KERNEL_THINNING_1 = [[-1, -1, -1], [0, 1, 0], [1, 1, 1]]
    KERNEL_THINNING_2 = [[0, -1, -1], [1, 1, -1], [1, 1, 0]]


class THRESHOLD_CONFIG:
    THR_BINARY = 'cv2.THRESH_BINARY'
    THR_BINARY_INV = 'cv2.THRESH_BINARY_INV'
    THR_TRUNC = 'cv2.THRESH_TRUNC'
    THR_TO_ZERO = 'cv2.THRESH_TOZERO'
    THR_TO_ZERO_INV = 'cv2.THRESH_TOZERO_INV'
    # use only with adaptive job
    THR_ADAPTIVE_MEAN_C = 'cv2.ADAPTIVE_THRESH_MEAN_C'
    THR_ADAPTIVE_GAUSS_C = 'cv2.ADAPTIVE_THRESH_GAUSSIAN_C'


# For dilating filters we used the following paper:
# https://www.researchgate.net/publication/343156325_Custom_Dilated_Edge_Detection_Filters


class FILTERS:
    # https://www.researchgate.net/publication/220695992_Machine_Perception_of_Three-Dimensional_Solids
    ROBERTS_2x2 = 'ROBERTS_2x2'
    # https://pdfs.semanticscholar.org/a9ea/aecf23f3a4b7822e4bcca924e02cd5b4dc4e.pdf
    PIXEL_DIFF_3x3 = 'PIXEL_DIFFERENCE_3x3'
    PIXEL_DIFF_5x5 = 'PIXEL_DIFFERENCE_DILATED_5x5'
    PIXEL_DIFF_7x7 = 'PIXEL_DIFFERENCE_DILATED_7x7'
    # https://pdfs.semanticscholar.org/a9ea/aecf23f3a4b7822e4bcca924e02cd5b4dc4e.pdf
    PIXEL_DIFF_SEPARATED_3x3 = 'SEPARATED_PIXEL_DIFFERENCE_3x3'
    PIXEL_DIFF_SEPARATED_5x5 = 'SEPARATED_PIXEL_DIFFERENCE_DILATED_5x5'
    PIXEL_DIFF_SEPARATED_7x7 = 'SEPARATED_PIXEL_DIFFERENCE_DILATED_7x7'
    # https://www.researchgate.net/publication/285159837_A_33_isotropic_gradient_operator_for_image_processing
    SOBEL_3x3 = 'SOBEL_3x3'
    SOBEL_DILATED_5x5 = 'SOBEL_DILATED_5x5'
    SOBEL_DILATED_7x7 = 'SOBEL_DILATED_7x7'
    # http://www.hlevkin.com/articles/SobelScharrGradients5x5.pdf
    # https://www.semanticscholar.org/paper/Sobel-Edge-Detection-Algorithm-Gupta-Mazumdar/6bcafdf33445585966ee6fb3371dd1ce15241a62
    # https://www.researchgate.net/publication/334001329_Expansion_and_Implementation_of_a_3x3_Sobel_and_Prewitt_Edge_Detection_Filter_to_a_5x5_Dimension_Filter
    # https://www.researchgate.net/publication/49619233_Image_Segmentation_using_Extended_Edge_Operator_for_Mammographic_Images
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.9053&rep=rep1&type=pdf
    SOBEL_5x5 = 'SOBEL_5x5'
    SOBEL_7x7 = 'SOBEL_7x7'
    # https://books.google.com/books?hl=en&lr=&id=vp-w_pC9JBAC&oi=fnd&pg=PA75&dq=Object+Enhancement+and+Extraction+Prewitt&ots=szJ83nmDF3&sig=OzsyzGtBeojFq2HHSsiVuJwNHHQ
    PREWITT_3x3 = 'PREWITT_3x3'
    PREWITT_DILATED_5x5 = 'PREWITT_DILATED_5x5'
    PREWITT_DILATED_7x7 = 'PREWITT_DILATED_7x7'
    # http://www.hlevkin.com/articles/SobelScharrGradients5x5.pdf
    PREWITT_5x5 = 'PREWITT_5x5'
    PREWITT_LEVKINE_5x5 = 'PREWITT_LEVKINE_5x5'
    PREWITT_7x7 = 'PREWITT_7x7'
    # https://ieeexplore.ieee.org/document/1674733
    FREI_CHEN_3x3 = 'FREI_CHEN_3x3'
    FREI_CHEN_5x5 = 'FREI_CHEN_DILATED_5x5'
    FREI_CHEN_7x7 = 'FREI_CHEN_DILATED_7x7'
    # https://www.researchgate.net/publication/36148383_Optimal_operators_in_digital_image_processing_Elektronische_Ressource
    SCHARR_3x3 = 'SCHARR_3x3'
    # http://www.hlevkin.com/articles/SobelScharrGradients5x5.pdf
    SCHARR_5x5 = 'SCHARR_5x5'
    # https://www.researchgate.net/publication/320011336_A_Novel_Region_Selection_Algorithm_for_Auto-focusing_Method_Based_on_Depth_from_Focus
    SCHARR_CHEN_5x5 = 'SCHARR_CHEN_5x5'
    SCHARR_DILATED_5x5 = 'SCHARR_DILATED_5x5'
    SCHARR_DILATED_7x7 = 'SCHARR_DILATED_7x7'
    # https://pubmed.ncbi.nlm.nih.gov/5562571/
    KIRSCH_3x3 = 'KIRSCH_3x3'
    # http://www.hlevkin.com/articles/SobelScharrGradients5x5.pdf
    KIRSCH_5x5 = 'KIRSCH_5x5'
    KIRSCH_DILATED_5x5 = 'KIRSCH_DILATED_5x5'
    KIRSCH_DILATED_7x7 = 'KIRSCH_DILATED_7x7'
    # https://www.researchgate.net/publication/272261227_Edge_Detection_on_Images_of_Pseudoimpedance_Section_Supported_by_Context_and_Adaptive_Transformation_Model_Images
    KAYYALI_3x3 = 'KAYYALI_3x3'
    KAYYALI_DILATED_5x5 = 'KAYYALI_DILATED_5x5'
    KAYYALI_DILATED_7x7 = 'KAYYALI_DILATED_7x7'
    # http://k-zone.nl/Kroon_DerivativePaper.pdf
    KROON_3x3 = 'KROON_3x3'
    KROON_DILATED_5x5 = 'KROON_DILATED_5x5'
    KROON_DILATED_7x7 = 'KROON_DILATED_7x7'
    # https://www.sciencedirect.com/science/article/abs/pii/S0734189X8980009X
    KITCHEN_MALIN_3x3 = 'KITCHEN_3x3'
    KITCHEN_MALIN_DILATED_5x5 = 'KITCHEN_DILATED_5x5'
    KITCHEN_MALIN_DILATED_7x7 = 'KITCHEN_DILATED_7x7'
    # https://www.researchgate.net/publication/281748455_A_New_Edge_Detection_Method_for_CT-Scan_Lung_Images
    # https://www.researchgate.net/publication/281443645_A_Novel_5x5_Edge_Detection_Operator_for_Blood_Vessel_Images
    EL_ARWADI_EL_ZAART_5x5 = 'EL_ARWADI_EL_ZAART_5x5'
    # https://www.springerprofessional.de/en/a-novel-edge-detection-operator-for-identifying-buildings-in-aug/18459582
    ORHEI_3x3 = 'ORHEI_3x3'
    ORHEI_5x5 = 'ORHEI_5x5'
    ORHEI_B_5x5 = 'ORHEI_V1_5x5'
    ORHEI_DILATED_5x5 = 'ORHEI_DILATED_5x5'
    ORHEI_DILATED_7x7 = 'ORHEI_DILATED_7x7'
    ORHEI_DILATED_9x9 = 'ORHEI_DILATED_9x9'
    # https://www.sciencedirect.com/science/article/abs/pii/S0146664X77800245
    ROBINSON_CROSS_3x3 = 'SOBEL_3x3'
    ROBINSON_CROSS_DILATED_5x5 = SOBEL_DILATED_5x5
    ROBINSON_CROSS_DILATED_7x7 = SOBEL_DILATED_7x7
    # https://www.sciencedirect.com/science/article/abs/pii/S0146664X77800245
    ROBINSON_MODIFIED_CROSS_3x3 = 'PREWITT_3x3'
    ROBINSON_MODIFIED_CROSS_5x5 = PREWITT_DILATED_5x5
    ROBINSON_MODIFIED_CROSS_7x7 = PREWITT_DILATED_7x7
    # https://www.sciencedirect.com/science/article/abs/pii/0010480971900346?via%3Dihub
    KIRSCH_CROSS_3x3 = 'KIRSCH_3x3'
    # https://books.google.ro/books?hl=en&lr=&id=vp-w_pC9JBAC&oi=fnd&pg=PA75&dq=%22Object+Enhancement+and+Extraction%22&ots=szJ80qlBD5&sig=5qT-zX7eoMUnEa4YiQ4wN9rPeDg&redir_esc=y#v=onepage&q=%22Object%20Enhancement%20and%20Extraction%22&f=false
    PREWITT_CROSS_3x3 = 'PREWITT_COMPASS_3x3'
    PREWITT_CROSS_DILATED_5x5 = 'PREWITT_COMPASS_DILATED_5x5'
    PREWITT_CROSS_DILATED_7x7 = 'PREWITT_COMPASS_DILATED_7x7'


class FILTERS_SECOND_ORDER:
    # https://www.sciencedirect.com/science/article/abs/pii/0734189X8990131X
    LAPLACE_1 = 'LAPLACE_V1_3x3'
    LAPLACE_5x5_1 = 'LAPLACE_V1_5x5'
    LAPLACE_DILATED_5x5_1 = 'LAPLACE_V1_DILATED_5x5'
    LAPLACE_DILATED_7x7_1 = 'LAPLACE_V1_DILATED_7x7'

    # http://fourier.eng.hmc.edu/e161/lectures/gradient/node7.html
    LAPLACE_2 = 'LAPLACE_V2_3x3'
    LAPLACE_5x5_2 = 'LAPLACE_V2_5x5'
    LAPLACE_DILATED_5x5_2 = 'LAPLACE_V2_DILATED_5x5'
    LAPLACE_DILATED_7x7_2 = 'LAPLACE_V2_DILATED_7x7'

    # https://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter5.pdf
    LAPLACE_3 = 'LAPLACE_V3_3x3'
    LAPLACE_DILATED_5x5_3 = 'LAPLACE_V3_DILATED_5x5'
    LAPLACE_DILATED_7x7_3 = 'LAPLACE_V3_DILATED_7x7'

    LAPLACE_4 = 'LAPLACE_V4_3x3'
    LAPLACE_DILATED_5x5_4 = 'LAPLACE_V4_DILATED_5x5'
    LAPLACE_DILATED_7x7_4 = 'LAPLACE_V4_DILATED_7x7'

    # https://www.sciencedirect.com/science/article/abs/pii/016516849290051W
    LAPLACE_5 = 'LAPLACE_V5_3x3'
    LAPLACE_DILATED_5x5_5 = 'LAPLACE_V5_DILATED_5x5'
    LAPLACE_DILATED_7x7_5 = 'LAPLACE_V5_DILATED_7x7'

    LOG_5x5_V1 = 'LOG_V1_5x5'
    LOG_7x7_V1 = 'LOG_V1_7x7'
    LOG_9x9_V1 = 'LOG_V1_9x9'


class CANNY_VARIANTS:
    MANUAL_THRESHOLD = None
    # https://www.cs.bgu.ac.il/~icbv161/wiki.files/Readings/1986-Canny-A_Computational_Approach_to_Edge_Detection.pdf
    FIX_THRESHOLD = 'CANNY_CONFIG.FIX_THRESHOLD'
    # https://ieeexplore.ieee.org/document/5739265
    RATIO_THRESHOLD = 'CANNY_CONFIG.RATIO_THRESHOLD'
    MEDIAN_SIGMA = 'CANNY_CONFIG.MEDIAN_SIGMA'
    RATIO_MEAN = 'CANNY_CONFIG.RATIO_MEAN'
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.403.5666&rep=rep1&type=pdf#page=120
    OTSU_HALF = 'CANNY_CONFIG.OTSU_HALF'
    # https://ieeexplore.ieee.org/abstract/document/5476095
    OTSU_MEDIAN_SIGMA = 'CANNY_CONFIG.OTSU_MEDIAN_SIGMA'


if __name__ == "__main__":
    pass
