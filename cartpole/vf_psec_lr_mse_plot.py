"""Plot value function results."""
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

PSEC_LRS = set()

def read_proto(filename):
    results = results_pb2.MethodResult()
    with open(filename, 'rb') as f:
        results.ParseFromString(f.read())
    return results


def plot_data(data, file_name, plot_params):
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)

    for method in sorted(data):
        label = method

        print ('method ' + method)

        sorted_trajs = sorted(data[method])
        x = []
        y = []
        ylower = []
        yupper = []
        for num_traj in sorted_trajs:
            sorted_lrs = sorted(data[method][num_traj])
            if 'PSEC' in method:
                for lr in sorted_lrs:
                    #if lr == 0.05:
                    #    continue
                    errors = np.array(data[method][num_traj][lr])

                    n = len(errors) # number of trials
                    mean = np.mean(errors)
                    std = np.std(errors)
                    #if 'Lin' in method:
                    print ('number of errors for {} trajs: {}, mean MSE {} std {} '.format(lr, len(data[method][num_traj][lr]), mean, 1.96 * std / np.sqrt(float(n))))
                    #if mean < 450 :
                    #    print (', '.join(map(str, errors)))
                    
                    #if mean >= 400:
                    #    continue
                    yerr = 1.96 * std / np.sqrt(float(n))
                    y.append(mean)
                    ylower.append(mean - yerr)
                    yupper.append(mean + yerr)
                    x.append(lr)
            else:
                x = list(PSEC_LRS)
                x = sorted(x)
                errors = np.array(data[method][num_traj][0])

                n = len(errors) # number of trials

                mean = np.mean(errors)
                std = np.std(errors)
                print ('number of errors for {} trajs: {}, MSE mean {}, std {}'.format(0, len(data[method][num_traj][0]), mean, 1.96 * std / np.sqrt(float(n))))
                #print (', '.join(map(str, errors)))
                yerr = 1.96 * std / np.sqrt(float(n))
                y = [mean for _ in range(len(x))]
                ylower = [mean - yerr for _ in range(len(x))]
                yupper = [mean + yerr for _ in range(len(x))]

            linestyle = '-'
            #linestyle = '-.'
            #linestyle = ':'
            if method == 'TD(0)':
                color = 'red'
                line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 5, color = color)
            else:
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

        #method_TD(0)_trial_847277_trajs_9_alpha_0.010000_tiling_20_t1_15_t2_9_psec-act_0_psec-nh_0_psec-ne_0_psec-lr_0.000000
        #method_TD(0)_trial_847277_trajs_9_alpha_0.010000_tiling_20_t1_15_t2_9_psec-act_0_psec-nh_0_psec-ne_0_psec-lr_0.000000
        #print (basename)
        specs = basename.split('_')
       
        # thing that is fixed is the seed and traj size, finding best performing based on these two fixed points
        # if td looking at lr, tc, etc
        # if psec, looking at above AND the extra PSEC stuff
 
        method = specs[specs.index('method') + 1]
        
        num_trajs = int(results.num_trajs) #int(results.num_steps_observed[0])

        if method not in data:
            data[method] = {}       
        if num_trajs not in data[method]:
            data[method][num_trajs] = {}

        psec_lr = float(specs[specs.index('psec-lr') + 1])

        if psec_lr not in data[method][num_trajs]:
            data[method][num_trajs][psec_lr] = []  

        errs = results.value_error
        data[method][num_trajs][psec_lr].append(errs)

        if 'PSEC' in method:
            PSEC_LRS.add(psec_lr)

    return data

def main():

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return

    data = collect_data() 

    plot_params = {'bfont': 40,
               'lfont': 35,
               'tfont': 40,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': None,
               'x_range': None,
               'log_scale': True,
               'x_label': 'PSEC Learning Rate',
               'y_label': 'MSVE',
               'shade_error': True,
               'x_mult': 1}
    plot_data(data, 'temp1', plot_params)
    
if __name__ == '__main__':
    main()
