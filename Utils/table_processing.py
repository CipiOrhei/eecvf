import config_main as CONFIG
from Utils.log_handler import log_util_info_to_console
import os

"""
Module handles the transition of data from code to latex table data of the EECVF
"""

def create_latex_cpm_table_list(variants: list, variants_public: list, sub_variants: list, sub_variants_pub: list,
                                operators: list, operators_pub: list, order: list, inputs: list, levels: list,
                                name_of_table: str, number_decimal: int = 3,
                                sub_var_dis: str =  'DILATED', print_to_console: bool = False, save_location: str = 'Logs/'):
    """
    :param variants: operator big variants
    :param variants_public: variants equivalent list to print
    :param sub_variants: operator variants
    :param sub_variants_pub: sub_variants equivalent list to print
    :param operators: list of operators, rows
    :param operators_pub: operators equivalent list to print
    :param number_decimal: number of decimal to take in consideration
    :param inputs: list of input ports
    :param levels: list of level input ports

    """
    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, 'PCM')
    d = dict()
    new_string = ''

    # get files from benchmark folder
    for dirname, dirnames, filenames in os.walk(input_location):
        for filename in filenames:
            # files.append(filename)
            f = open(os.path.join(input_location, filename)).readlines()[-1].split('   ')
            d[filename.split('.')[0]] = {
                # 'f1': round(float(f[3]), number_decimal),
                # 'r': round(float(f[1]), number_decimal),
                # 'p': round(float(f[2]), number_decimal)
                'f1': format(float(f[3]), '.' + str(number_decimal) + 'f'),
                'r': format(float(f[1]), '.' + str(number_decimal) + 'f'),
                'p': format(float(f[2]), '.' + str(number_decimal) + 'f'),
            }



    print(d)

    table = '\\begin{tabular}{|l|c|'
    # add column per variant and sub-variant
    for i in range(len(variants)):
        new_text = 'c' * len(sub_variants[i]) + '|'
        table += new_text
    # add horizontal lines
    table += '}\n\hline\n\hline\n'

    # add header
    table += '\\multicolumn{2}{|c|}{\\bfseries Operator} '
    for i in range(len(variants)):
        new_text = '&\\multicolumn{' + str(len(sub_variants[i])) + '}{c|}{\\bfseries ' + variants_public[i] + '}'
        table += new_text
    table += '\\\\\n\hline\n'

    # add second header
    table += '\t&'
    for i in range(len(variants)):
        new_text = ''
        for v in sub_variants_pub[i]:
            new_text += '\t&' + v
        table += new_text
    table += '\\\\\n\hline\n'

    for op_idx in range(len(operators)):
        line_1 = '\t&' + order[0]
        line_2 = operators_pub[op_idx] + '\t&' + order[1]
        line_3 = '\t&' + order[2]
        for var_idx in range(len(variants)):
            for sub_var_idx in range(len(sub_variants[var_idx])):
                # source = '{var}_{op}_{sub_var}_{inp}_{lvl}'.format(var=variants[var_idx], op=operators[op_idx],
                #                                                    sub_var=sub_variants[var_idx][sub_var_idx], inp=inputs[var_idx], lvl=levels[var_idx])
                tmp = ''
                for id in d.keys():
                    if (variants[var_idx] in id) and (operators[op_idx] in id) and (sub_variants[var_idx][sub_var_idx] in id) and \
                            (inputs[var_idx] in id) and (levels[var_idx] in id) \
                            and ((sub_var_dis in sub_variants[var_idx][sub_var_idx] and sub_var_dis in id) or
                            (sub_var_dis not in sub_variants[var_idx][sub_var_idx] and sub_var_dis not in id)):
                        tmp = id

                if tmp != '':
                    line_1 += '\t&' + str(d[tmp][order[0].lower()])
                    line_2 += '\t&' + str(d[tmp][order[1].lower()])
                    line_3 += '\t&' + str(d[tmp][order[2].lower()])
                else:
                    line_1 += '\t&-'
                    line_2 += '\t&-'
                    line_3 += '\t&-'


        line_1 += '\\\\\n'
        line_2 += '\\\\\n'
        line_3 += '\\\\\n\\hline\n'

        table += line_1 + line_2 + line_3


    table += '\n\hline\n\\end{tabular}}'
    print(table)

    if print_to_console is True:
        log_util_info_to_console(table)
    # save table
    file_to_save = open(os.path.join(save_location, name_of_table + '.txt'), 'w')
    file_to_save.write(table)
    file_to_save.close()


def create_latex_cpm_table(header_list: list, list_of_data: list, name_of_table: str,
                           prefix_data_name: str = None, suffix_data_name: str = None, level_data_name: str = 'L0',
                           version_data_name: list = None, version_separation=None, list_of_series=None,
                           data_per_variant: list = ['P', 'R', 'F1'], number_decimal: int = 3, print_to_console: bool = False,
                           save_location: str = 'Logs/'):
    """
    Create a latex table for CPM data. It uses as input *.log file generated from Benchmark.BDSD module.

    Data:
    FINAL_SOBEL_DILATED_7x7_BLURED_L0.log, FINAL_SCHARR_DILATED_5x5_BLURED_L0.log so on
    Input:
    header_list=['Variant', '', '3x3', '5x5', 'Dilated 5x5', '7x7', 'Dilated 7x7'],
    prefix_data_name='FINAL', suffix_data_name='BLURED', level_data_name='L0',
    version_data_name=['3x3', '5x5', 'DILATED_5x5', '7x7', 'DILATED_7x7'], data_per_variant=['R', 'P', 'F1']
    Example:
    Output:
    &Variant	&	&3x3	&5x5	&Dilated 5x5	&7x7	&Dilated 7x7	\\
    \hline
    &	        &R	&0.599	&0.581	&0.581	&0.566	&0.566\\
    &PIXEL	    &P	&0.604	&0.614	&0.614	&0.619	&0.619\\
    &	        &F1	&0.602	&0.597	&0.597	&0.591	&0.591\\

    :param header_list: list of string that represent the header of the table
    :param list_of_data: data to be used in filling the table
    :param name_of_table: name for the table
    :param prefix_data_name: prefix of series name
    :param suffix_data_name: suffix of series name
    :param level_data_name: level of series name
    :param version_data_name: versions of the series to use as columns
    :param data_per_variant: order of R, P, F1 lines per version
    :param number_decimal: number of decimals for data representation
    :param save_location: where to save the plot
    :param print_to_console: if we want the output to be printed to console
    :return None
    """
    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, 'PCM')
    d = dict()
    new_string = ''

    # get files from benchmark folder
    for dirname, dirnames, filenames in os.walk(input_location):
        for filename in filenames:
            # files.append(filename)
            f = open(os.path.join(input_location, filename)).readlines()[-1].split('   ')
            d[filename.split('.')[0]] = {'f1': round(float(f[3]), number_decimal),
                                         'r': round(float(f[1]), number_decimal),
                                         'p': round(float(f[2]), number_decimal)}

    table = ""
    # create header of the table
    for el in header_list:
        table += "&" + el + "\t"
    else:
        table += "\\\\\n"
        table += "\\hline\n"

    series = list()
    if list_of_series is None:
        # get unique series in the list of data
        for data in list_of_data:
            t = (((data.split(level_data_name)[0]).split(suffix_data_name)[0]).split(prefix_data_name)[-1]).split('_')[1]
            if t not in series:
                series.append(t)
    else:
        series.extend(list_of_series)

    dict_keys = d.keys()

    for serie in series:
        for data in data_per_variant:
            # make sure to write series name in middle of the R-P-F1 lines
            if data == data_per_variant[len(data_per_variant) // 2]:
                line = "&" + serie + "\t&" + data
            else:
                line = "&\t&" + data
            # find correct series data from data dictionary
            for ver in version_data_name:
                data_to_add = None
                for key in dict_keys:
                    if (serie in key) and (ver in key) and (prefix_data_name in key) and (suffix_data_name in key) and  (level_data_name in key) \
                            and (version_separation is None or ((version_separation in key and version_separation in ver) or
                                                                (version_separation not in key and version_separation not in ver))):
                        data_to_add = key
                        break

                if data_to_add is not None:
                    line += "\t&" + str(d[data_to_add][data.lower()])
                else:
                    line += "\t&-"
            table += line + "\\\\\n"
        table += "\\hline\n"
    # print table
    if print_to_console is True:
        log_util_info_to_console(table)
    # save table
    file_to_save = open(os.path.join(save_location, name_of_table + '.txt'), 'w')
    file_to_save.write(table)
    file_to_save.close()


def create_latex_fom_table(number_decimal: int = 3, order_by: bool = True, number_of_series: int = 25, data = 'FOM'):
    input_location = os.path.join(CONFIG.BENCHMARK_RESULTS, data)
    d = dict()
    new_string = ''

    # get files from benchmark folder
    for dirname, dirnames, filenames in os.walk(input_location):
        for filename in filenames:
            # files.append(filename)
            try:
                f = open(os.path.join(input_location, filename)).readlines()[-1].split(' ')

                for idx in range(len(f), -1, -1):
                    value = f[idx - 1]
                    if value != '' and value != '\n':
                        break

                d[filename.split('.')[0]] = round(float(value), number_decimal)
            except:
                print(filename)

    # print(d)

    if order_by is True:
        d = (sorted(d.items(), key=lambda key_value: key_value[1], reverse=True))[:number_of_series]

    for el in d:
        print(el, '\n')

if __name__ == "__main__":
    pass
