import sys
import os
import cv2
# noinspection PyUnresolvedReferences
import config_main as CONFIG
import glob

"""
These wrapper helps the to run multiple main_TMBuD_detection.py variants without memory contamination of BOF from opencv
"""

desc_list = [cv2.AKAZE_DESCRIPTOR_KAZE]
diff_list = [cv2.KAZE_DIFF_PM_G1]
desc_size_list = [8]
nOctaves_list = [6]
nLayes_list = [3]
thr_list = [0.8]
thr_akaze_list = [0.0010]
dictionarySize_list = [450]
kernel_smoothing = [CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_7x7_1]
smoothing_strength = [0.9]
use_gps = False
distance_list = [100]

total = len(desc_list) * len(diff_list) * len(desc_size_list) * len(nOctaves_list) * len(nLayes_list) * len(thr_list) * len(dictionarySize_list) * len(kernel_smoothing) * len(smoothing_strength) * len(distance_list) * len(thr_akaze_list)
idx = 0
print('Current iteration: ', idx, '\\', total)
for desc in desc_list:
    for diff in diff_list:
        for desc_size in desc_size_list:
            for nOctaves in nOctaves_list:
                for nLayes in nLayes_list:
                    for thr in thr_list:
                        for thr_akaze in thr_akaze_list:
                            for dict_size in dictionarySize_list:
                                for kernel in kernel_smoothing:
                                    for st_kernel in smoothing_strength:
                                        for dist_w in distance_list:
                                            # os.system("python main_TMBuD_detection.py create_bow " + str(desc) + "\t" + str(diff) + "\t" + str(desc_size) + "\t" + str(nOctaves) + "\t" + str(nLayes) + "\t" + str(thr) + "\t" + str(thr_akaze) +
                                            #           "\t" + str(dict_size) + "\t" + str(kernel) + "\t" + str(st_kernel) + "\t" + str(use_gps) + "\t" + str(dist_w))

                                            os.system("python main_TMBuD_detection.py inquiry " + str(desc) + "\t" + str(diff) + "\t" + str(desc_size) + "\t" + str(nOctaves) + "\t" + str(nLayes) + "\t" + str(thr) + "\t" + str(thr_akaze) +
                                                      "\t" + str(dict_size) + "\t" + str(kernel) + "\t" + str(st_kernel) + "\t" + str(use_gps) + "\t" + str(dist_w))

                                            idx +=1

                                            print('Current iteration: ', idx, '\\', total)

dict_results = dict()
# change this folder for your use case
list_files = glob.glob(r'c:\repos\eecvf_git\Logs\benchmark_results\ZuBuD Top1\*.log')

for file in list_files:
    with open(file) as f:
        content = f.readlines()[-1].split(' ')
        dict_results[content[-3]] = float(content[-2])

dict_results = sorted(dict_results.items(), key=lambda kv: kv[1], reverse=False)

for el in dict_results:
    print(el)
