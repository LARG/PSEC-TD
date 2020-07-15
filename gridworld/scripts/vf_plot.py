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
    #fig.set_size_inches(13.5, 12.0, forward=True)
    fig.set_size_inches(16.25, 12.5, forward=True)

    for method in data:
        label = method
        sorted_trajs = sorted(data[method])
        x = []
        y = []
        ylower = []
        yupper = []
        for num_traj in sorted_trajs:
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
        if 'TD-Estimate' in label:
            linestyle = '-.'
        line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 5)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)

    if plot_params['log_scale']:
        ax.set_xscale('log')
        #ax.set_yscale('log')
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

def main():

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return

    data = {}
    visited_data = {}
    deterministic_probs = {}
    max_ind = 0
    best = 0.0

    for basename in os.listdir(FLAGS.result_directory):
        filename = os.path.join(FLAGS.result_directory, basename)
        if '.err' in filename:
            continue
        try:
            results = read_proto(filename)
        except Exception as e:
            if 'Tag' not in str(e):
                raise e

        method = basename.split('_')[1]
        trial = int(basename.split('_')[3])
        num_trajs = int(results.num_steps_observed[0])
        err = results.value_error[0]
        unvisited_ratio = results.num_unvisited_s_a[0]
        deterministic_prob = results.deterministic_prob[0]
        if method not in data:
            data[method] = {}
            visited_data[method] = {}
            deterministic_probs[method] = {}
        if num_trajs not in data[method]:
            data[method][num_trajs] = []
            visited_data[method][num_trajs] = []
            deterministic_probs[method][num_trajs] = []
        if len(results.value_error) == 0:
            continue

        data[method][num_trajs].append(err)
        visited_data[method][num_trajs].append(unvisited_ratio)
        deterministic_probs[method][num_trajs].append(deterministic_prob)
        print ('method {}, num_trajs {}, err: {}, unvisited_ratio {}'.format(method, num_trajs, err, unvisited_ratio))

    plot_params = {'bfont': 45,
               'lfont': 40,
               'tfont': 45,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': None,
               'x_range': None,
               'log_scale': True,
               'x_label': 'Number of Episodes (m)',
               'y_label': 'MSVE (vs. PSEC-MDP-CEE)',
               'shade_error': True,
               'x_mult': 1}
    plot_data(data, 'PSEC-CEE', plot_params)

    plot_params['legend'] = None
    plot_params['y_label'] = 'Fraction of Unvisited (s,a,s\')'
    plot_data(visited_data, 'unvisited_fraction', plot_params)

    #plot_data(deterministic_probs, 'temp3', plot_params)
   
if __name__ == '__main__':
    main()

