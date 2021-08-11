# noinspection PyPep8Naming
import config_main as CONFIG
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from Utils.log_handler import log_error_to_console

"""
Module handles the plotting for the benchmarking the EECVF
"""

# noinspection SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection
my_colors = [
    'dimgray', 'darkgrey', 'silver', 'rosybrown', 'lightcoral', 'indianred', 'firebrick', 'red', 'orangered', 'darkorange', 'lightsalmon',
    'sienna', 'sandybrown', 'peru', 'wheat', 'darkgoldenrod', 'goldenrod', 'gold', 'darkkhaki', 'olive', 'lightsalmon', 'yellow',
    'yellowgreen', 'olivedrab', 'lawngreen', 'darkseagreen', 'forestgreen', 'darkgreen', 'seagreen', 'springgreen', 'mediumaquamarine',
    'turquoise', 'paleturquoise', 'lightseagreen', 'darkslategray', 'teal', 'darkturquoise', 'cadetblue', 'deepskyblue', 'lightskyblue',
    'steelblue', 'slategrey', 'royalblue', 'midnightblue', 'blue', 'slateblue', 'mediumpurple', 'rebeccapurple', 'indigo', 'darkviolet',
    'thistle', 'violet', 'magenta', 'deeppink', 'hotpink', 'palevioletred', 'crimson', 'pink', 'blueviolet', 'fuchsia', 'tomato', 'grey',
    'linen', 'deepskyblue', 'darkorchid', 'deeppink', 'bisque', 'chocolate', 'navajowhite', 'gainsboro','linen', 'peru', 'gray'
]

color_list = list(set(my_colors))


# noinspection SpellCheckingInspection
def plot_f1_score_cpm(series_name: str, x_label: str, x_name: str, y_label: str, y_name: list, plot_names: list,
                      variants_of_series: list = None, title: str = None, add_series: bool = False, name: str = None,
                      save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = False):
    """
    Plot f1 score accordingly to series that changes.
    Example:
        series_name='FINAL', x_label='S', y_label='K', x_name='Sigma',
        y_name=['F1', 'P'], plot_names=['F1-measure', 'Precision']
        Will plot the results of FINAL_xxx_S_xx_K_xx series for F1 and Precision
    :param series_name: series name
    :param x_label: label in name for the series for x axis
    :param x_name: what that label represents-> for legend details
    :param y_label: label in name for the series for y axis
    :param y_name: what that label represents-> for legend details
    :param title: Title of the plot
    :param plot_names: Name of plots in the legend
    :param add_series: add series name to plot name
    :param series_name: series name
    :param variants_of_series: variants of the series
    :param name: name to save
    :param save_location: where to save the plot
    :param show_plot: shot the plot
    :param save_plot: save the plot
    :return None
    """
    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, 'PCM')
    files = []
    d = dict()
    series_available = []
    series = []
    x_min = y_min = 100
    x_max = y_max = 0

    if variants_of_series is not None:
        for s in variants_of_series:
            series.append(series_name + '_' + s)
    else:
        series.append(series_name)

    # get files from benchmark folder
    for s in series:
        for dir_name, dir_names, filenames in os.walk(input_location):
            for filename in filenames:
                if s in filename:
                    files.append(filename)
                    t = filename.split(y_label + '_')[1]
                    t = t.split('_')[0]
                    series_available.append(t)

    for s in series:
        colors = iter(random.sample(color_list, k=len(set(series_available))))

        # noinspection SpellCheckingInspection
        for serie in list(set(series_available)):
            for file in files:
                if s + '_' + y_label + '_' + serie in file:
                    f = open(os.path.join(input_location, file))
                    tmp = f.readlines()[-1]
                    tmp = tmp.split('   ')
                    d[float(file.split(x_label + '_')[-1].split('_')[0])] = {'f1': float(tmp[3]), 'r': float(tmp[1]), 'p': float(tmp[2])}
                    # print(tmp)
                    f.close()

            x = list(d.keys())
            x.sort()

            if x_min > x[0]:
                x_min = x[0]

            if x_max < x[-1]:
                x_max = x[-1]

            color = next(colors)

            for data_name, plot_name in zip(y_name, plot_names):
                res = []
                for el in x:
                    t = d[el][data_name.lower()]
                    res.append(t)

                    if y_min > t:
                        y_min = t

                    if y_max < t:
                        y_max = t

                if data_name == 'F1':
                    ch = (0, ())
                elif data_name == 'P':
                    ch = (0, (1, 1))
                else:
                    ch = (0, (3, 5, 1, 5, 1, 5))

                if add_series:
                    if len(s) == 1:
                        plt.plot(x, res, label=plot_name + ' ' + y_label + '_' + serie, linestyle=ch, color=color)
                    else:
                        plt.plot(x, res, label=plot_name + ' ' + y_label + '_' + serie + s.split(series_name)[1], linestyle=ch, color=color)
                else:
                    plt.plot(x, res, label=plot_name, linestyle=ch, color=color)

    fig = plt.gcf()
    fig.set_size_inches(w=15, h=10)
    plt.xlabel(xlabel=x_name)
    # plt.ylabel(y_name)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(ticks=np.arange(0, x_max * 1.1, x_max / 20))
    plt.yticks(ticks=np.arange(y_min * 0.95, y_max * 1.05, y_max / 40))
    plt.legend(fancybox=True, fontsize='small', loc='best')

    if title is not None:
        plt.title(title)

    if save_plot is True:
        if name is None:
            plt.savefig(os.path.join(save_location, '{}.png'.format('f1_plot_' + series_name)), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_location, '{}.png'.format(name)), bbox_inches='tight')

    if show_plot is True:
        plt.show()

    plt.close()


# noinspection SpellCheckingInspection
def plot_cpm_results(list_of_data: list, inputs: list, prefix: str = '', level: str = 'L0', number_decimal: int = 3,
                     show_input: bool = False, show_level: bool = False, order_by_f1: bool = False, name: str = 'PCM_results',
                     save_location: str = 'Logs/', show_plot: bool = False, save_plot: bool = False):
    """
    Plot f1-r-p diagram of series.
    [PREFIX]_series_[INPUT]_[LEVEL]
    :param list_of_data: data to plot
    :param inputs: inputs of series name
    :param prefix: prefix of series name
    :param level: level of data to plot
    :param number_decimal: number of decimals
    :param show_input: add input to series name
    :param show_level: add level to series name
    :param order_by_f1: order the plots by the f1 score
    :param name: name to save
    :param save_location: where to save the plot
    :param show_plot: shot the plot
    :param save_plot: save the plot
    :return None
    """

    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, 'PCM')
    d = dict()
    # noinspection PyBroadException
    try:
        colors = iter(random.sample(color_list, k=len(set(list_of_data))))
    except BaseException:
        log_error_to_console('PLOT CPM ERROR', 'REDUCE THE NUMBER OF SERIES')
        return

    # get files from benchmark folder
    for dir_name, dir_names, filenames in os.walk(input_location):
        for filename in filenames:
            f = open(os.path.join(input_location, filename)).readlines()[-1].split('   ')
            d[filename.split('.')[0]] = {'f1': round(float(f[3]), number_decimal),
                                         'r': round(float(f[1]), number_decimal),
                                         'p': round(float(f[2]), number_decimal)}

    fig = plt.figure()
    space = np.linspace(start=0.01, stop=1.00, num=100, retstep=True)
    p, r = np.meshgrid(space[0], space[0])
    f = 2 * np.array(p) * np.array(r) / (np.array(p) + np.array(r))
    plt.contour(p, r, f, levels=10)
    plt.set_cmap(cmap='Greens')
    plt.box(on=True)
    plt.grid(b=True, linestyle='--')
    plt.xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.xlabel(xlabel='Recall')
    plt.ylabel(ylabel='Precision')
    plt.title(label='F1-measure')

    to_plot = []
    for data in list_of_data:
        for inp in inputs:
            color = next(colors)
            tmp = prefix + '_' + data + '_' + inp.upper() + '_' + level
            tmp_label = data

            if show_input is True:
                tmp_label += '_' + inp.upper()

            if show_level is True:
                tmp_label += '_' + level

            # noinspection PyBroadException
            try:
                if order_by_f1 is False:
                    plt.plot(d[tmp]['r'], d[tmp]['p'], label=tmp_label, marker='o', color=color)
                else:
                    to_plot.append((d[tmp]['r'], d[tmp]['p'], tmp_label, color, d[tmp]['f1']))
            except BaseException:
                log_error_to_console('PLOT CPM ERROR', 'Data not available: {}'.format(tmp_label))

    if order_by_f1 is True:
        to_plot = sorted(to_plot, key=lambda key_value: key_value[4], reverse=True)
        for i in to_plot:
            # "F={value} {name}".format(value=str(i[4]),name=i[2])
            plt.plot(i[0], i[1], label="[F={value:0.3f}] {name}".format(value=i[4], name=i[2]), marker='o', color=i[3])

    fig.set_size_inches(w=15, h=15)
    plt.legend(fancybox=True, fontsize='small', loc='best')
    # plt.legend(fancybox=True, fontsize='small', title='Legend', loc='right', bbox_to_anchor=(fig_size/2,fig_size/2,2,2))

    if save_plot is True:
        plt.savefig(os.path.join(save_location, '{}.jpg'.format(name)), bbox_inches='tight')

    if show_plot is True:
        plt.show()

    plt.close()


def plot_first_cpm_results(list_of_data: list, number_of_series: int, inputs: list = [''], prefix: str = '', level: str = 'L0',
                           self_contained_list: bool = False, number_decimal: int = 3, show_input: bool = False, show_level: bool = False,
                           order_by: str = None, name: str = 'PCM_results', save_location: str = 'Logs/',
                           replace_list: list = None, font_size_labels: int = 14,
                           prefix_to_cut_legend = None, suffix_to_cut_legend=None, set_legend_left = False, set_all_to_legend = False,
                           show_plot: bool = False, save_plot: bool = False):
    """
    Plot f1-r-p diagram of series.
    [PREFIX]_series_[INPUT]_[LEVEL]
    :param list_of_data: data to plot
    :param inputs: inputs of series name
    :param number_of_series: inputs of series name
    :param prefix: prefix of series name
    :param replace_list: list of strings to replace in legend
    :param level: level of data to plot
    :param number_decimal: number of decimals
    :param set_all_to_legend: set data of F1, P, and R to plot
    :param self_contained_list: if name is in data
    :param show_input: add input to series name
    :param show_level: add level to series name
    :param order_by: order the plots by the f1 score
    :param font_size_labels: size of label font
    :param prefix_to_cut_legend: string to cut at de beginning
    :param suffix_to_cut_legend: string to cut at de ending
    :param set_legend_left: set legend to the left of the image
    :param set_all_to_legend: set R-P-F1 to legend
    :param name: name to save
    :param save_location: where to save the plot
    :param show_plot: shot the plot
    :param save_plot: save the plot
    :return None
    """

    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, 'PCM')
    d = dict()
    # noinspection PyBroadException
    try:
        colors = iter(random.sample(color_list, k=number_of_series))
    except BaseException:
        log_error_to_console('PLOT CPM ERROR', 'REDUCE THE NUMBER OF SERIES')
        return

    # get files from benchmark folder
    for dir_name, dir_names, filenames in os.walk(input_location):
        for filename in filenames:
            f = open(os.path.join(input_location, filename)).readlines()[-1].split('   ')
            d[filename.split('.log')[0]] = {'f1': round(float(f[3]), number_decimal),
                                            'r': round(float(f[1]), number_decimal),
                                            'p': round(float(f[2]), number_decimal)}

    fig = plt.figure()
    space = np.linspace(start=0.01, stop=1.00, num=100, retstep=True)
    p, r = np.meshgrid(space[0], space[0])
    f = 2 * np.array(p) * np.array(r) / (np.array(p) + np.array(r))
    plt.contour(p, r, f, levels=10)
    plt.set_cmap('Greens')
    plt.box(on=True)
    plt.grid(b=True, linestyle='--')
    plt.xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=14)
    plt.yticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=14)
    plt.ylim(ymin=0)  # this line
    plt.xlim(xmin=0)  # this line
    plt.xlabel(xlabel='Recall', fontsize=font_size_labels)
    plt.ylabel(ylabel='Precision', fontsize=font_size_labels)
    plt.title(label='F1-measure', fontsize=font_size_labels)

    to_plot = []
    for data in list_of_data:
        for inp in inputs:
            if self_contained_list is False:
                tmp = prefix + '_' + data + '_' + inp.upper() + '_' + level
            else:
                tmp = data
            tmp_label = data

            if show_input is True:
                tmp_label += '_' + inp.upper()

            if show_level is True:
                tmp_label += '_' + level

            # noinspection PyBroadException
            try:
                to_plot.append((d[tmp]['r'], d[tmp]['p'], tmp_label, d[tmp]['f1']))
            except BaseException:
                log_error_to_console('PLOT CPM ERROR', 'Data not available: {}'.format(tmp_label))

    ds = {'f1': 3, 'r': 0, 'p': 1}
    if order_by is not None:
        to_plot = (sorted(to_plot, key=lambda key_value: key_value[ds[order_by]], reverse=True))[:number_of_series]
    for i in to_plot:
        # "F={value} {name}".format(value=str(i[4]),name=i[2])
        name_legend = i[2]
        if suffix_to_cut_legend is not None or prefix_to_cut_legend is not None:
            name_legend = (name_legend.split(suffix_to_cut_legend)[0]).split(prefix_to_cut_legend)[-1]

        if replace_list is not None:
            for el in replace_list:
                name_legend = name_legend.replace(el[0], el[1])

        color = next(colors)

        if set_all_to_legend is False:
            plt.plot(i[0], i[1], label="[F={value:0.3f}] {name}".format(value=i[3], name=name_legend), marker='o', color=color)
        else:

            plt.plot(i[0], i[1], label="[F1={f1:0.3f}][P={p:0.3f}][R={r:0.3f}]{name}".format(f1=i[3], p=i[1], r=i[0], name=name_legend), marker='o', color=color)

    fig.set_size_inches(15, 15)

    if set_legend_left is True:
        plt.legend(fancybox=True, fontsize='x-large', loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(fancybox=True, fontsize='x-large', loc='best')

    if save_plot is True:
        plt.savefig(os.path.join(save_location, '{}.jpg'.format(name)), bbox_inches='tight')

    if show_plot is True:
        plt.show()

    plt.close()


if __name__ == "__main__":
    CONFIG.BENCHMARK_RESULTS = r'C:\repos\eecvf\Logs\benchmark_results'
    # plot_f1_score_CPM(series_name='FINAL', x_label='S', y_label='K', x_name='Sigma', y_name='P', title='Kernel size', show_plot=True)
    # plot_cpm_results(['THN_THR_SOBEL_DILATED_7x7_MEDIAN_GREY_L0', 'THN_THR_SOBEL_DILATED_7x7_GREY_L0'])
    pass
