import argparse
from protos import results_pb2
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
from matplotlib import pyplot as plt

from matplotlib import rcParams
import os


def bool_argument(parser, name, default=False, msg=''):
    dest = name.replace('-', '_')
    parser.add_argument('--%s' % name, dest=dest, type=bool, default=default, help=msg)
    parser.add_argument('--no-%s' % name, dest=dest, type=bool, default=default, help=msg)

parser = argparse.ArgumentParser()
parser.add_argument('result_directory', help=help)
FLAGS = parser.parse_args()

rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

def read_proto(filename):
    results = results_pb2.MethodResult()
    with open(filename, 'rb') as f:
        results.ParseFromString(f.read())
    return results


def plot_data(data, file_name, plot_params):
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)

    for method in sorted(data):
        '''
        action-value namings
        if 'EXP' in method:
            label = 'EXP-AV-TD'
        elif 'PSEC' in method:
            label = 'PSEC-AV-TD'
        else:
            label = 'AV-TD'
        '''

        label = method
        if 'PSEC' in method:
            continue
            if 'Estimate' in method:
                label = 'PSEC-TD(0)'
            else:
                continue

        sorted_trajs = sorted(data[method])
        x = []
        y = []
        ylower = []
        yupper = []
        for num_traj in sorted_trajs:
            print ('number of errors for {} trajs: {}'.format(num_traj, len(data[method][num_traj])))
            errors = np.array(data[method][num_traj])
            n = len(errors) # number of trials
            mean = np.mean(errors)
            std = np.std(errors)

            yerr = 1.96 * std / np.sqrt(float(n))
            y.append(mean)
            ylower.append(mean - yerr)
            yupper.append(mean + yerr)
            x.append(num_traj)

        linestyle = '-'
        if 'PSEC-TD' in label:
            linestyle = ':'
        if 'PSEC-TD-Estimate' in label:
            linestyle = '-.'
        if method == 'TD(0)' or method == 'LSTD':
            line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 5, color = 'red')
        else:
            line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 5)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha, linestyle = linestyle, linewidth = 5)

    if plot_params['log_scale']:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if plot_params['x_range'] is not None:
        plt.xlim(plot_params['x_range'])
    if plot_params['y_range'] is not None:
        plt.ylim(plot_params['y_range'])

    ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

    ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'])
    ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'])

    if plot_params['legend']:
        plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                   ncol=plot_params['legend_cols'])
    fig.tight_layout()
    plt.savefig('{}.pdf'.format(file_name))
    plt.close()
    #plt.show()

def collect_data():

    data = {}

    td_method = 'td-0-vs-mdp'
    ristd_method = 'ris-' + td_method

    for basename in os.listdir(FLAGS.result_directory):
        filename = os.path.join(FLAGS.result_directory, basename)
        if '.err' in filename:
            continue
        try:
            results = read_proto(filename)
        except Exception as e:
            if 'Tag' not in str(e):
                raise e

        specs = basename.split('_')

        if 'alpha' not in basename:
            continue
        method = specs[specs.index('method') + 1]
        lr = float(specs[specs.index('alpha') + 1])
        num_trajs = int(results.num_steps_observed[0])

        if method not in data:
            data[method] = {}       
        if num_trajs not in data[method]:
            data[method][num_trajs] = {}
        if lr not in data[method][num_trajs]:
            # list of final errors across all trials for a given batch size and lr
            data[method][num_trajs][lr] = []

        errs = results.value_error
        data[method][num_trajs][lr].append(errs)

    return data


def best_lr(data):

    best_data = {}
    for method in data:

        best_data[method] = {}
        for num_traj in data[method]:
            
            min_err = float('inf')
            min_err_lr = -2
            for lr in data[method][num_traj]:
                errs = np.array(data[method][num_traj][lr])
                mean = np.mean(errs)
                print (mean)
                if mean < min_err:
                    min_err = mean
                    min_err_lr = lr
            if min_err_lr == -2:
                continue
            print ('best lr method: {} num traj: {} lr: {} error: {}'.format(method, num_traj, min_err_lr, min_err))
            best_data[method][num_traj] = data[method][num_traj][min_err_lr]
    return best_data 

def main():

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return

    data = collect_data() 

    best_lr_data = best_lr(data)

    plot_params = {'bfont': 45,
               'lfont': 35,
               'tfont': 45,
               'legend': True,
               'legend_loc': 3,
               'legend_cols': 2,
               'y_range': (None, None),
               'x_range': None,
               'log_scale': True,
               'x_label': 'Number of Episodes (m)',
               'y_label': 'MSVE',
               'shade_error': True,
               'x_mult': 1}
    plot_data(best_lr_data, 'param_sweep_result', plot_params)
    
if __name__ == '__main__':
    main()
