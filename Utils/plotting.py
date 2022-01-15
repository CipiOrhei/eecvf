# noinspection PyPep8Naming
import random

import config_main as CONFIG
import matplotlib.pyplot as plt
import numpy as np
import os
from operator import add
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Utils.plot_benchmark_results import color_list

"""
Module handles the plotting for the EECVF
"""


def get_table_data_from_csv(content: str, table_number: int) -> dict:
    """
    Get data from csv file produced by the APPL block into a dictionary.
    :param content: csv data
    :param table_number: csv can have multiple tables inside
    :return: dictionary with the data
    """
    tables = []
    for el in content.split('Frame'):
        if el is not '':
            tables.append('Frame' + el)
    # create dict to hold data
    table_dict = {k: [] for k in (tables[table_number - 1].split(',False\n'))[0].split(',')}
    keys_dict = list(table_dict.keys())

    for element in (tables[table_number - 1].split(',False\n'))[1:]:
        data = element.split(',')
        if data != ['']:
            for index in range(len(keys_dict)):
                table_dict[keys_dict[index]].append(data[index])

    return table_dict


def plot_avg_time_jobs(input_location: str = CONFIG.LOG_KPI_FILE, table_number: int = 1, save_location: str = 'Logs/',
                       x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=25, y_ticks_font_size=25, dpi_save_value=300,
                       title_font_size=35, img_size_w=15, img_size_h=10, legend_font_size='small', legend_name='Jobs', title_name=None,
                       show_plot: bool = False, save_plot: bool = False, eliminate_get_image=False, show_legend=True) -> None:
    """
    Plot average time of jobs.
    :param input_location location of input csv files
    :param table_number if there are more than one application run in one main
    :param show_plot if we want to show the plot
    :param save_plot if we want to save the plot
    :param show_legend add legend to plot
    :param x_label_font_size font size of x axis label
    :param y_label_font_size font size of y axis label
    :param x_ticks_font_size font size of x axis ticks
    :param y_ticks_font_size font size of y axis ticks
    :param title_font_size font size of plot title
    :param title_name name of plot title
    :param legend_font_size size of legend font (xx-small, x-small, small, medium, large, x-large, xx-large)
    :param legend_name name of legend
    :param img_size_w save width image
    :param img_size_h save height image
    :param dpi_save_value save dpi of image
            The bigger value the longer time it takes to save
    :param eliminate_get_image if we don't want the get image job plotted
    :param save_location: location to save the plots
    :return None
    """
    file = open(input_location, 'r')
    table_dict = get_table_data_from_csv(file.read(), table_number)

    if eliminate_get_image is True:
        table_dict.pop('Get frame Avg Time[ms]')

    # get only keys regarding average time
    keys = [el for el in list(table_dict.keys()) if '[ms]' in el]

    max_value = 0
    data = list(map(float, table_dict['Frame']))

    for element in range(len(keys)):
        series = list(map(float, table_dict[keys[element]]))
        if max_value < max(series):
            max_value = max(series)
        plt.plot(series, data=data, label=keys[element].split('Avg Time[ms]')[0])

    fig = plt.gcf()
    fig.set_size_inches(w=img_size_w, h=img_size_h)
    plt.xlabel('Frames', fontsize=x_label_font_size)
    plt.ylabel('Avg time [ms]', fontsize=y_label_font_size)
    plt.xlim(int(table_dict['Frame'][0]), int(table_dict['Frame'][-1]))
    plt.yticks(np.arange(0, max_value * 1.1, max_value / 20), fontsize=y_ticks_font_size)
    plt.xticks(np.arange(0, data[-1], round(data[-1], -1) // 20), fontsize=x_ticks_font_size)
    plt.ylim(0)
    if title_name is not None:
        plt.title(title_name, fontsize=title_font_size)
    if show_legend is True:
        plt.legend(fancybox=True, fontsize=legend_font_size, title=legend_name, loc='best')

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, 'avg_time_plot.png'), bbox_inches='tight', dpi=dpi_save_value)

    plt.clf()
    plt.close()
    file.close()


def plot_custom_list(port_list: list, name_to_save: str, input_location: str = CONFIG.LOG_KPI_FILE, table_number: int = 1, set_frame_name: bool = False,
                     x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=25, y_ticks_font_size=25, dpi_save_value=300,
                     title_font_size=35, img_size_w=15, img_size_h=10, legend_font_size='small', legend_name=None, title_name=None,
                     save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = True, y_plot_name: str = 'Avg time [ms]', set_name_replace_list=None):
    """
      Plot custom ports
        :param port_list: list of ports to plot
        :param name_to_save: name you want for plot
        :param input_location: location of input data csv
        :param set_frame_name: use name of image instead of frame number on x axis
        :param set_name_replace_list: list of string to replace for labels in legend
        :param y_plot_name: name of y label
        :param x_label_font_size font size of x axis label
        :param y_label_font_size font size of y axis label
        :param x_ticks_font_size font size of x axis ticks
        :param y_ticks_font_size font size of y axis ticks
        :param title_font_size font size of plot title
        :param title_name name of plot title
        :param legend_font_size size of legend font (xx-small, x-small, small, medium, large, x-large, xx-large)
        :param legend_name name of legend
        :param img_size_w save width image
        :param img_size_h save height image
        :param dpi_save_value save dpi of image
                The bigger value the longer time it takes to save
       :param show_plot: if we want to show the plot
       :param save_plot: if we want to save the plot
       :param table_number: what table from csv to plot
       :param save_location: where to save
       :return None
      """
    file = open(input_location, 'r')
    table_dict = get_table_data_from_csv(file.read(), table_number)

    keys = []

    for port in port_list:
        table_keys = list(table_dict.keys())
        for element in table_keys:
            if port in element:
                keys.append(element)

    # get only keys regarding average time
    # keys = [el for el in list(table_dict.keys()) if '[ms]' in el]

    max_value = 0
    if set_frame_name is False:
        data = list(map(float, table_dict['Frame']))
    else:
        for el in range(len(table_dict['Image Name'])):
            table_dict['Image Name'][el] = table_dict['Image Name'][el].split('.')[0]
        data = list(map(str, table_dict['Image Name']))

    for element in range(len(keys)):
        series = list(map(float, table_dict[keys[element]]))
        if max_value < max(series):
            max_value = max(series)

        set_name = keys[element].split('Avg Time[ms]')[0]
        if set_name_replace_list is not None:
            for el in set_name_replace_list:
                set_name = set_name.replace(el[0], el[1])

        plt.plot(data, series, label=set_name)

    fig = plt.gcf()
    fig.set_size_inches(w=img_size_w, h=img_size_h)
    plt.xlabel('Frames', fontsize=x_label_font_size)
    plt.ylabel(y_plot_name, fontsize=y_label_font_size)

    if set_frame_name is False:
        plt.xlim(int(table_dict['Frame'][0]), int(table_dict['Frame'][-1]))

    plt.ylim(0)

    if legend_name is not None:
        plt.legend(fancybox=True, fontsize=legend_font_size, loc='best', title=legend_name)
    else:
        plt.legend(fancybox=True, fontsize=legend_font_size, loc='best')

    plt.yticks(fontsize=y_ticks_font_size)
    plt.xticks(fontsize=x_ticks_font_size)

    if title_name is not None:
        plt.title(title_name, fontsize=title_font_size)

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, name_to_save + '.png'), bbox_inches='tight', dpi=dpi_save_value)

    plt.clf()
    plt.close()
    file.close()


def plot_GLCM_data(port_list: list, caracteristic: str, title: str,
                   input_location: str = CONFIG.LOG_KPI_FILE, table_number: int = 1, name_to_save: str = None,
                   x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=25, y_ticks_font_size=25, dpi_save_value=300,
                   title_font_size=35, img_size_w=15, img_size_h=10, legend_font_size='small', legend_name='Jobs', title_name=None,
                   save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = True, y_plot_name: str = None):
    """
      Plot custom ports
      :param port_list: list of ports to plot
      :param name_to_save: name you want for plot
      :param input_location: location of input data csv
      :param caracteristic: characteristic to plot
      :param y_plot_name: name of y label
      :param x_label_font_size font size of x axis label
      :param y_label_font_size font size of y axis label
      :param x_ticks_font_size font size of x axis ticks
      :param y_ticks_font_size font size of y axis ticks
      :param title_font_size font size of plot title
      :param title_name name of plot title
      :param legend_font_size size of legend font (xx-small, x-small, small, medium, large, x-large, xx-large)
      :param legend_name name of legend
      :param img_size_w save width image
      :param img_size_h save height image
      :param dpi_save_value save dpi of image
              The bigger value the longer time it takes to save
      :param show_plot: if we want to show the plot
      :param save_plot: if we want to save the plot
      :param table_number: what table from csv to plot
      :param save_location: where to save
      :return None
      """
    file = open(input_location, 'r')
    table_dict = get_table_data_from_csv(file.read(), table_number)

    keys = []

    if caracteristic not in ['CONTRAST', 'DISSIMILARITY', 'HOMOGENEITY', 'ASM', 'ENERGY', 'CORRELATION', 'ENTROPY']:
        return

    if name_to_save is None:
        name_to_save = title + '_' + caracteristic.lower()

    if y_plot_name is None:
        y_plot_name = caracteristic

    port_list_new = list()
    for el in range(len(port_list)):
        port_list_new.append('GLCM ' + caracteristic + ' ' + port_list[el])

    for port in port_list_new:
        table_keys = list(table_dict.keys())
        for element in table_keys:
            if port in element:
                keys.append(element)

    # get only keys regarding average time
    # keys = [el for el in list(table_dict.keys()) if '[ms]' in el]

    max_value = 0
    data = list(map(float, table_dict['Frame']))

    for element in range(len(keys)):
        series = list(map(float, table_dict[keys[element]]))
        if max_value < max(series):
            max_value = max(series)
        label_name = keys[element].split('Avg Time[ms]')[0]
        label_name = label_name.replace('GLCM ', '').replace('GLCM_', '').replace('_LC0', '')
        label_name = label_name.replace('CONTRAST ', '').replace('DISSIMILARITY ', '').replace('HOMOGENEITY ', '').replace('ASM ', '').replace('ENERGY ', '').replace('CORRELATION ', '').replace('ENTROPY ', '')
        label_name = label_name.replace(np.pi.__str__().replace('.', '_'), '$\pi$').replace((np.pi/2).__str__().replace('.', '_'), '$\pi$/2').replace((np.pi/4).__str__().replace('.', '_'), '$\pi$/4')
        label_name = label_name.replace('D_', 'D=').replace('A_', 'A=').replace('_', ' ')
        plt.plot(series, data=data, label=label_name)

    fig = plt.gcf()
    fig.set_size_inches(w=img_size_w, h=img_size_h)
    plt.xlabel('Frames', fontsize=x_label_font_size)
    plt.ylabel(y_plot_name, fontsize=y_label_font_size)
    plt.xlim(int(table_dict['Frame'][0]), int(table_dict['Frame'][-1]))
    plt.yticks(fontsize=y_ticks_font_size)
    plt.xticks(fontsize=x_ticks_font_size)
    plt.ylim(0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=len(keys), fontsize=legend_font_size, title=legend_name)

    if title_name is not None:
        plt.title(title_name, fontsize=title_font_size)

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, name_to_save + '.png'), bbox_inches='tight', dpi=dpi_save_value)

    plt.clf()
    plt.close()
    file.close()


# noinspection PyUnusedLocal
def plot_time_jobs(port_list: list, series_names: list, name_to_save: str, input_location: str = CONFIG.LOG_KPI_FILE, table_number: int = 1,
                   x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=25, y_ticks_font_size=25, dpi_save_value=300,
                   title_font_size=35, img_size_w=15, img_size_h=10, legend_font_size='small', legend_name='Jobs', title_name=None,
                   save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = True):
    """
      Plot custom average time of jobs
      :param port_list: list of ports to plot
      :param series_names: list of series names for plot
      :param name_to_save: name you want for plot
      :param input_location: location of input data csv
        :param x_label_font_size font size of x axis label
        :param y_label_font_size font size of y axis label
        :param x_ticks_font_size font size of x axis ticks
        :param y_ticks_font_size font size of y axis ticks
        :param title_font_size font size of plot title
        :param title_name name of plot title
        :param legend_font_size size of legend font (xx-small, x-small, small, medium, large, x-large, xx-large)
        :param legend_name name of legend
        :param img_size_w save width image
        :param img_size_h save height image
        :param dpi_save_value save dpi of image
                The bigger value the longer time it takes to save
      :param show_plot: if we want to show the plot
      :param save_plot: if we want to save the plot
      :param table_number: what table from csv to plot
      :param save_location: where to save
      :return None
      """
    file = open(input_location, 'r')
    table_dict = get_table_data_from_csv(file.read(), table_number)

    new_dict = dict()

    table_keys = list(table_dict.keys())
    for index in range(len(series_names)):
        new_dict[series_names[index]] = [0 for el in range(len(table_dict[table_keys[0]]))]
        for element in port_list[index]:
            for el in table_keys:
                if element in el:
                    new_dict[series_names[index]] = list(map(add, new_dict[series_names[index]], list(map(float, table_dict[el]))))
                    break

    max_value = 0
    data = list(map(float, table_dict['Frame']))

    for element in new_dict.keys():
        series = list(map(float, new_dict[element]))
        if max_value < max(series):
            max_value = max(series)
        plt.plot(series, data=data, label=element.split('Avg Time[ms]')[0])

    fig = plt.gcf()
    fig.set_size_inches(w=img_size_w, h=img_size_h)
    plt.xlabel('Frames', fontsize=x_label_font_size)
    plt.ylabel('Avg time [ms]', fontsize=y_label_font_size)
    plt.xlim(int(table_dict['Frame'][0]), int(table_dict['Frame'][-1]))
    plt.yticks(np.arange(0, max_value * 1.1, max_value / 20), fontsize=y_ticks_font_size)
    plt.xticks(np.arange(0, data[-1], round(data[-1], -1) // 20), fontsize=x_ticks_font_size)
    plt.ylim(0)
    plt.legend(fancybox=True, fontsize=legend_font_size, title=legend_name, loc='best')

    if title_name is not None:
        plt.title(title_name, fontsize=title_font_size)

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, name_to_save + '.png'), bbox_inches='tight', dpi=dpi_save_value)

    plt.clf()
    plt.close()
    file.close()


def plot_box_benchmark_values(name_to_save: str, eval: list,
                              number_decimal: int = 3, number_of_series: int = None, data='FOM', data_subsets=None,
                              x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=25, y_ticks_font_size=25, dpi_save_value=300,
                              title_font_size=35, img_size_w=15, img_size_h=10, legend_font_size='small', legend_name='Jobs', title_name=None,
                              save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = True):
    """
    Plot average time of jobs.
    :param name_to_save name for file to save
    :param eval list of port evaluation to include in plot
    :param number_decimal number of decimal to use for values
    :param number_of_series number of series
    :param data name of eval category to plot
    :param data_subsets subset of data to plot
    :param show_plot if we want to show the plot
    :param save_plot if we want to save the plot
    :param show_legend add legend to plot
    :param x_label_font_size font size of x axis label
    :param y_label_font_size font size of y axis label
    :param x_ticks_font_size font size of x axis ticks
    :param y_ticks_font_size font size of y axis ticks
    :param title_font_size font size of plot title
    :param title_name name of plot title
    :param legend_font_size size of legend font (xx-small, x-small, small, medium, large, x-large, xx-large)
    :param legend_name name of legend
    :param img_size_w save width image
    :param img_size_h save height image
    :param dpi_save_value save dpi of image
            The bigger value the longer time it takes to save
    :param save_location: location to save the plots
    :return None
    """

    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, data)
    subset_dict = dict()

    for data_subset in data_subsets:
        subset_dict[data_subset] = {}

    # get files from benchmark folder
    for dirname, dirnames, filenames in os.walk(input_location):
        for filename in filenames:
            # files.append(filename)
            try:
                f = open(os.path.join(input_location, filename)).readlines()[-1].split(' ')

                for idx in range(len(f), 0, -1):
                    value = f[idx - 1]
                    if value != '' and value != '\n':
                        break

                for data_subset in data_subsets:
                    fxx = filename.split('.')[0]
                    if (data_subset in filename) and (fxx in eval):
                        subset_dict[data_subset][filename.split('.')[0]] = round(float(value), number_decimal)

            except:
                print(filename)

    for data_subset in subset_dict:
        if number_of_series is None:
            subset_dict[data_subset] = (sorted(subset_dict[data_subset].items(), key=lambda key_value: key_value[1], reverse=True))
        else:
            subset_dict[data_subset] = (sorted(subset_dict[data_subset].items(), key=lambda key_value: key_value[1], reverse=True))[:number_of_series]
        # print(subset_dict[data_subset])

    for set in subset_dict:
        if 'RDE' in data:
            print('min', subset_dict[set][-1])
        else:
            print('max', subset_dict[set][0])

    subset_values_dict = {}

    for data_subset in subset_dict:
        subset_values_dict[data_subset] = [variant[1] for variant in subset_dict[data_subset]]

    dataframe = pd.DataFrame.from_dict(subset_values_dict)
    dataframe.set_index(dataframe.columns[0])
    colors = ['blue', 'green', 'purple', 'tan', 'pink', 'red']

    df = dataframe.plot(kind='box',
                        labels=list(subset_dict.keys()),
                        figsize=(img_size_w, img_size_h), showmeans=True, grid=True)

    plt.ylabel(data, fontsize=y_label_font_size)

    # colors = iter(random.sample(color_list, k=len(subset_dict)))
    #
    # for set in subset_dict:
    #     # print(subset_dict[set][0])
    #     color = next(colors)
    #     plt.plot(label="[{data}={val:0.3f}]{name}".format(data=data, val=subset_dict[set][0][1], name=subset_dict[set][0][0]), marker='o', color=color)
    #     plt.legend(fancybox=True, fontsize='x-large', loc='center left', bbox_to_anchor=(1, 0.5))

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, name_to_save + '.png'), bbox_inches='tight', dpi=dpi_save_value)

    plt.clf()
    plt.close()


def plot_histogram_grey_image(image, name_folder, picture_name,
                              x_label_font_size=30, y_label_font_size=30, x_ticks_font_size=20, y_ticks_font_size=20, dpi_save_value=300,
                              title_font_size=30, img_size_w=15, img_size_h=10,
                              to_show=False, to_save=False):
    """
    Plots the histogram of an image.
    :param image: the image we wish to process the histogram on. Make sure it is grey.
    :param name_folder: folder name where to output it
    :param picture_name: name of picture. it will be used in the save name.
    :param x_label_font_size font size of x axis label
    :param y_label_font_size font size of y axis label
    :param x_ticks_font_size font size of x axis ticks
    :param y_ticks_font_size font size of y axis ticks
    :param title_font_size font size of plot title
    :param img_size_w save width image
    :param img_size_h save height image
    :param dpi_save_value save dpi of image
            The bigger value the longer time it takes to save
    :param to_show: if we desire to show the histogram on run time
    :param to_save: if we desire to save the histogram
    :return: None
    """
    fig = plt.gcf()
    fig.set_size_inches(w=img_size_w, h=img_size_h)
    # plt.hist(image.flatten(), bins=256, range=[0, 256], color='b')
    histogram, bin_edges = np.histogram(image, bins=256, range=[0, 256])
    plt.plot(bin_edges[0:-1], histogram)
    plt.xlim([0, 256])
    plt.xlabel('Pixel value', fontsize=x_label_font_size)
    plt.ylabel('Pixel number', fontsize=y_label_font_size)
    # plt.legend(fancybox=True, fontsize='small', loc='best')
    plt.xticks(fontsize=x_ticks_font_size)  # fontsize of the tick labels
    plt.yticks(fontsize=y_ticks_font_size)  # fontsize of the tick labels
    plt.title(picture_name, fontsize=title_font_size)
    file_to_save = os.path.join(CONFIG.APPL_SAVE_LOCATION, name_folder)
    if not os.path.exists(file_to_save):
        os.makedirs(file_to_save)
    if to_save:
        plt.savefig(os.path.join(file_to_save, '{}.png'.format('hist_' + picture_name)), bbox_inches='tight', dpi=dpi_save_value)
    if to_show:
        plt.show()

    plt.clf()
    plt.close()


def plot_histogram_rgb_image(image, name_folder, picture_name,
                             x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=25, y_ticks_font_size=25, dpi_save_value=300,
                             title_font_size=35, img_size_w=15, img_size_h=10,
                             to_show=False, to_save=False):
    """
    Plots the histogram of an image.
    :param image: the image we wish to process the histogram on. Make sure it is grey.
    :param name_folder: folder name where to output it
    :param x_label_font_size font size of x axis label
    :param y_label_font_size font size of y axis label
    :param x_ticks_font_size font size of x axis ticks
    :param y_ticks_font_size font size of y axis ticks
    :param title_font_size font size of plot title
    :param img_size_h save height image
    :param dpi_save_value save dpi of image
            The bigger value the longer time it takes to save
    :param picture_name: name of picture. it will be used in the save name.
    :param to_show: if we desire to show the histogram on run time
    :param to_save: if we desire to save the histogram
    :return: None
    """
    fig = plt.gcf()
    fig.set_size_inches(w=img_size_w, h=img_size_h)
    # tuple to select colors of each channel line
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=256, range=(0, 256) )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    # plt.hist(image.flatten(), bins=256, range=[0, 256], color='b')
    plt.xlim([0, 256])
    plt.xlabel('Pixel value', fontsize=x_label_font_size)
    plt.ylabel('Pixel number', fontsize=y_label_font_size)
    plt.xticks(fontsize=x_ticks_font_size)  # fontsize of the tick labels
    plt.yticks(fontsize=y_ticks_font_size)  # fontsize of the tick labels

    plt.title(picture_name, fontsize=title_font_size)
    file_to_save = os.path.join(CONFIG.APPL_SAVE_LOCATION, name_folder)
    if not os.path.exists(file_to_save):
        os.makedirs(file_to_save)

    if to_save:
        plt.savefig(os.path.join(file_to_save, '{}.png'.format('hist_' + picture_name)), bbox_inches='tight', dpi=dpi_save_value)

    if to_show:
        plt.show()

    plt.clf()
    plt.close()


def plot_frame_values(name_to_save: str, eval: list, data,
                      number_decimal: int = 3, set_name_replace_list=None,
                      x_label_font_size=25, y_label_font_size=25, x_ticks_font_size=25, y_ticks_font_size=25, dpi_save_value=300,
                      title_font_size=35, img_size_w=15, img_size_h=10, legend_font_size='small', legend_name='Jobs', title_name=None,
                      save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = True):
    """
      Plot custom ports
        :param name_to_save: name you want for plot
        :param eval: list of ports evaluated to plot
        :param number_decimal: number of decimals to use
        :param set_name_replace_list: list of string to replace for labels in legend
        :param y_plot_name: name of y label
        :param x_label_font_size font size of x axis label
        :param y_label_font_size font size of y axis label
        :param x_ticks_font_size font size of x axis ticks
        :param y_ticks_font_size font size of y axis ticks
        :param title_font_size font size of plot title
        :param title_name name of plot title
        :param legend_font_size size of legend font (xx-small, x-small, small, medium, large, x-large, xx-large)
        :param legend_name name of legend
        :param img_size_w save width image
        :param img_size_h save height image
        :param dpi_save_value save dpi of image
                The bigger value the longer time it takes to save
       :param show_plot: if we want to show the plot
       :param save_plot: if we want to save the plot
       :param save_location: where to save
       :return None
      """
    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, data)
    subset_dict = dict()

    # get files from benchmark folder
    for dirname, dirnames, filenames in os.walk(input_location):
        for filename in filenames:
            # files.append(filename)
            try:
            # if True:
                f = open(os.path.join(input_location, filename)).readlines()
                set_name = filename.split('.')[0]

                if set_name in eval:

                    if set_name_replace_list is not None:
                        for el in set_name_replace_list:
                            set_name = set_name.replace(el[0], el[1])

                    subset_dict[set_name] = {'average': 0, 'frames': list(), 'values': list()}

                    for line in range(len(f)):
                        if line == 0:
                            pass
                        elif line == (len(f)-1):
                            tmp = f[line].split(' ')
                            for el in range(len(tmp)-1, 0, -1):
                                if tmp[el] != '' and tmp[el] != ' ' and tmp[el] != '\n':
                                    subset_dict[set_name]['average'] = round(float(tmp[el]), number_decimal)
                                    break

                        else:
                            tmp = f[line].split(' ')
                            subset_dict[set_name]['frames'].append(tmp[0])
                            for el in range(len(tmp)-1, 0, -1):
                                if tmp[el] != '' and tmp[el] != ' ' and tmp[el] != '\n':
                                    subset_dict[set_name]['values'].append(round(float(tmp[el]), number_decimal))
                                    break

            except:
                print(filename, 'NOK TO USE FOR PLOTTING')

    for set in subset_dict.keys():
        plt.plot(subset_dict[set]['frames'], subset_dict[set]['values'], label=set)

    fig = plt.gcf()
    fig.set_size_inches(w=img_size_w, h=img_size_h)
    plt.xlabel('Frames', fontsize=x_label_font_size)
    plt.ylabel(data, fontsize=y_label_font_size)
    plt.yticks(fontsize=y_ticks_font_size)
    plt.xticks(fontsize=x_ticks_font_size)
    plt.legend(fancybox=True, fontsize=legend_font_size, loc='best', title=legend_name)

    if title_name is not None:
        plt.title(title_name, fontsize=title_font_size)

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, name_to_save + '.png'), bbox_inches='tight', dpi=dpi_save_value)

    plt.clf()
    plt.close()


if __name__ == "__main__":
    plot_GLCM_data(port_list=['GLCM_D_1_A_0_LC0', 'GLCM_D_1_A_3_141592653589793_LC0', 'GLCM_D_1_A_0_7853981633974483_LC0',
                              'GLCM_D_2_A_0_LC0', 'GLCM_D_2_A_3_141592653589793_LC0', 'GLCM_D_2_A_3_141592653589793_LC0',
                              'GLCM_D_4_A_0_LC0', 'GLCM_D_4_A_3_141592653589793_LC0', 'GLCM_D_4_A_3_141592653589793_LC0',
                              ],
                   # y_plot_name='HOMOGENEITY',
                   caracteristic='CONTRAST',
                   title='649f510',
                   # name_to_save='test',
                   input_location=r'c:\repos\eecvf_git\Logs\log.csv',
                   save_location=r'c:\repos\eecvf_git\Logs')
