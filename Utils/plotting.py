# noinspection PyPep8Naming
import config_main as CONFIG
import matplotlib.pyplot as plt
import numpy as np
import os
from operator import add

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
                       show_plot: bool = False, save_plot: bool = False, eliminate_get_image=False, show_legend=True) -> None:
    """
    Plot average time of jobs.
    :param input_location location of input csv files
    :param table_number if there are more than one application run in one main
    :param show_plot if we want to show the plot
    :param save_plot if we want to save the plot
    :param show_legend add legend to plot
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
    fig.set_size_inches(15, 10)
    plt.xlabel('Frames')
    plt.ylabel('Avg time [ms]')
    plt.xlim(int(table_dict['Frame'][0]), int(table_dict['Frame'][-1]))
    plt.yticks(np.arange(0, max_value * 1.1, max_value / 20))
    plt.xticks(np.arange(0, data[-1], round(data[-1], -1) // 20))
    plt.ylim(0)
    if show_legend is True:
        plt.legend(fancybox=True, fontsize='small', title='Jobs', loc='best')

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, 'avg_time_plot.jpg'))

    file.close()


def plot_custom_list(port_list: list, name_to_save: str, input_location: str = CONFIG.LOG_KPI_FILE, table_number: int = 1,
                     save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = True, y_plot_name: str = 'Avg time [ms]'):
    """
      Plot custom ports
      :param port_list: list of ports to plot
      :param name_to_save: name you want for plot
      :param input_location: location of input data csv
      :param y_plot_name: name of y label
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
    data = list(map(float, table_dict['Frame']))

    for element in range(len(keys)):
        series = list(map(float, table_dict[keys[element]]))
        if max_value < max(series):
            max_value = max(series)
        plt.plot(series, data=data, label=keys[element].split('Avg Time[ms]')[0])

    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.xlabel('Frames', fontsize=18)
    plt.ylabel(y_plot_name, fontsize=18)
    plt.xlim(int(table_dict['Frame'][0]), int(table_dict['Frame'][-1]))
    # plt.yticks(np.arange(0, max_value * 1.1, max_value / 20))
    # plt.xticks(np.arange(0, data[-1], round(data[-1], -1) // 20))
    plt.ylim(0)
    plt.legend(fancybox=True, fontsize='small', title='Jobs', loc='best')

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, name_to_save + '.jpg'))

    plt.close()
    file.close()


# noinspection PyUnusedLocal
def plot_time_jobs(port_list: list, series_names: list, name_to_save: str, input_location: str = CONFIG.LOG_KPI_FILE, table_number: int = 1,
                   save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = True):
    """
      Plot custom average time of jobs
      :param port_list: list of ports to plot
      :param series_names: list of series names for plot
      :param name_to_save: name you want for plot
      :param input_location: location of input data csv
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
    fig.set_size_inches(15, 10)
    plt.xlabel('Frames')
    plt.ylabel('Avg time [ms]')
    plt.xlim(int(table_dict['Frame'][0]), int(table_dict['Frame'][-1]))
    plt.yticks(np.arange(0, max_value * 1.1, max_value / 20))
    plt.xticks(np.arange(0, data[-1], round(data[-1], -1) // 20))
    plt.ylim(0)
    plt.legend(fancybox=True, fontsize='small', title='Jobs', loc='best')

    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(os.path.join(save_location, name_to_save + '.jpg'))

    plt.close()
    file.close()


if __name__ == "__main__":
    pass
