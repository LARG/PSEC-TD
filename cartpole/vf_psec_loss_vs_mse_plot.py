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

def read_proto(filename):
    results = results_pb2.MethodResult()
    with open(filename, 'rb') as f:
        results.ParseFromString(f.read())
    return results

def plot_losses(data, file_name, plot_params):
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)


    for method in data:
        tr_losses = np.array(data[method]['tr'])

        val_losses = np.array(data[method]['val'])
        tr_mean = np.mean(tr_losses, axis = 0)[:250]
        tr_std = np.std(tr_losses, axis = 0)[:250]
        val_mean = np.mean(val_losses, axis = 0)[:250]
        val_std = np.std(val_losses, axis = 0)[:250]
        n = len(tr_mean) # number of trials
        x = [i * 10 for i in range(n)]

        if 'Lin' in method:
            print ('tr avged vs. steps {}'.format(tr_mean))
            print ('val avged vs. steps {}'.format(val_mean))
       
        yerr = 1.96 * tr_std / np.sqrt(float(n))
        ylower = tr_mean - yerr
        yupper = tr_mean + yerr

        linestyle = '-'
        line, = plt.plot(x, tr_mean, label=method+'-tr', linestyle = linestyle, linewidth = 5)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)

        yerr = 1.96 * val_std / np.sqrt(float(n))
        ylower = val_mean - yerr
        yupper = val_mean + yerr
        
        linestyle = '-.'
        line, = plt.plot(x, val_mean, label=method+'-val', linestyle = linestyle, linewidth = 5)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)

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

def plot_mses(data, file_name, plot_params):
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)

    for method in data:
        mses = np.array(data[method]['mse'])

        mean = np.mean(mses, axis = 0)[:250]
        std = np.std(mses, axis = 0)[:250]
        n = len(mean) # number of trials
        x = [i * 10 for i in range(n)]
        #mean = mean[2:]
        #std = std[2:]
        #x = x[:-2]
        #print ('mse avged vs. steps {}'.format(mean))

        yerr = 1.96 * std / np.sqrt(float(n))
        ylower = mean - yerr
        yupper = mean + yerr

        linestyle = '-'
        line, = plt.plot(x, mean, label=method, linestyle = linestyle, linewidth = 5)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)

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
    c = 0
    td_method = 'td-0-vs-mdp'
    ristd_method = 'ris-' + td_method

    tr_losses = []
    val_losses = []
    mses = []

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
       
        print (method)

        if method not in data:
            data[method] = {}
            data[method]['tr'] = []
            data[method]['val']  = []
            data[method]['mse'] = []
        #print (results.psec_tr_loss)
        #print (results.per_value_error)
        #print (results.value_error)
        #print ('---')
        data[method]['tr'].append(results.psec_tr_loss)
        data[method]['val'].append(results.psec_val_loss)
        data[method]['mse'].append(results.per_value_error)


    #for t,v,m in zip(tr_losses, val_losses, mses):
    #    print (t,v,m)
    return data#tr_losses, val_losses, mses

def main():

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return


    #tr_losses, val_losses, mses = collect_data() 
    data = collect_data() 
    #data = []
    #data.append(tr_losses)
    #data.append(val_losses)
    #data.append(mses)

    #tfont 35
    plot_params = {'bfont': 40,
               'lfont': 25,
               'tfont': 40,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': (0.3, 0.6),
               'x_range': None,
               'log_scale': False,
               'x_label': 'epochs',
               'y_label': 'loss',
               'shade_error': True,
               'x_mult': 1}
   
    plot_losses(data, 'losses', plot_params)
    plot_params = {'bfont': 40,
               'lfont': 25,
               'tfont': 40,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': (50, 250),
               'x_range': None,
               'log_scale': False,
               'x_label': 'epochs',
               'y_label': 'MSVE',
               'shade_error': True,
               'x_mult': 1}


    plot_mses(data, 'mses', plot_params)
    
if __name__ == '__main__':
    main()
