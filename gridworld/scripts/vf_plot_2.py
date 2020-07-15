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


def plot_data(data, file_name, plot_params):
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)

    for method in data:
        label = method
        print ('method ' + method)

        sorted_trajs = sorted(data[method])
        if len(data[method]) == 0:
            continue
        x = []
        y = []
        ylower = []
        yupper = []
        for num_traj in sorted_trajs:
            #print ('number of errors for {} trajs: {}'.format(num_traj, len(data[method][num_traj])))
            errors = np.array(data[method][num_traj])
            n = len(errors) # number of trials
            mean = np.mean(errors)
            std = np.std(errors)

            yerr = 1.96 * std / np.sqrt(float(n))
            y.append(mean)
            ylower.append(mean - yerr)
            yupper.append(mean + yerr)
            x.append(num_traj)

        #x = np.arange(len(y))
        line, = plt.plot(x, y, label=label)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, ylower, yupper, facecolor=color, alpha=alpha)

        ''' 
        mean = np.mean(np.array(data[method]), axis=0)
        std = np.std(np.array(data[method]), axis=0)

        x = np.arange(np.size(mean)) * plot_params['x_mult']
        n = np.size(data[method], axis=0)
        print (method, n)

        yerr = 1.96 * std / np.sqrt(float(n))

        line, = plt.plot(x, mean, label=label)
        color = line.get_color()
        alpha = 0.5
        plt.fill_between(x, mean - yerr, mean + yerr, facecolor=color, alpha=alpha)
        '''

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

    for basename in os.listdir(FLAGS.result_directory):
        filename = os.path.join(FLAGS.result_directory, basename)
        if '.err' in filename:
            continue
        try:
            results = read_proto(filename)
        except Exception as e:
            if 'Tag' not in str(e):
                raise e

        # method_td-0-vs-mdp_trial_748070_steps_1000000_deterministic_1.0
        #method = '_'.join(basename.split('_')[:-1])
        #method_RISTD(0)-RIS-True-type-single-k-None-bufftype-None_trial_307734_steps_1000000_deterministic_1.0_lr_0.010000
        specs = basename.split('_')

        method = specs[specs.index('method') + 1]
        lr = specs[specs.index('alpha') + 1]

        if method not in data:
            data[method] = {}
        if lr not in data[method]:
            data[method][lr] = {}
            data[method][lr]['x'] = []
            data[method][lr]['y'] = []

        errs = results.batch_process_mses
        steps = results.batch_process_num_steps
        data[method][lr]['x'].append(steps)
        data[method][lr]['y'].append(errs)

    return data


def lr_plot(data, param_plot_params, best_plot_params, measure, learning_curve):

    '''
    Need to categorize based on specific method and learning rate
    For each method and lr, you have n trials
    each trial is error vs t.
    take the avg (like above) across trials per time-step

    then at this point, for a single method and learning rate, we have
    a single array of error vs time (averaged from above)
    
    if measure is perf
        look at the last 200 steps
    else if measure is AUC
        look at the last ALL steps

    average them, and plot that single value for this

    for this specific method, plot this single value for this particular lr

    for this specific method, repeat for all lr

    at each point, find method and lr combo that results in that min single value

    plot this graph (with error vs time, for this method/lr) wth confidence intervals for
    the n trials

    '''
    measure_plot_data = {}
    best_plot_data = {}

    for method in data:

        if method not in measure_plot_data:
            measure_plot_data[method] = {}

        for lr in data[method]:
            errs = np.array(data[method][lr]['y'])
            print (len(errs))
            print (len(errs[0]))
            print (len(errs[1]))
            print (len(errs[2]))
            mean = np.mean(errs, axis = 0)
            
            if measure == 'perf':
                val = np.mean(mean[-200:])
            elif measure == 'AUC':
                val = np.mean(mean)

            measure_plot_data[method][lr] = val

    for method in measure_plot_data:
        min_val = float('inf')
        min_lr = None
        
        if method not in best_plot_data:
            best_plot_data[method] = {}
        
        for lr in measure_plot_data[method]:
            val = measure_plot_data[method][lr]
            if val <= min_val:
                min_val = val
                min_lr = lr
        
        if min_lr not in best_plot_data[method]:
            best_plot_data[method][min_lr] = {}
            best_plot_data[method][min_lr]['x'] = []
            best_plot_data[method][min_lr]['y'] = []

        best_plot_data[method][min_lr]['x'] = data[method][min_lr]['x'] # returns 2d array
        best_plot_data[method][min_lr]['y'] = data[method][min_lr]['y'] # returns 2d array
        if '50' in method:
            print (len(data[method][min_lr]['x'][0]))

    #plot_param(measure_plot_data, 'lr_{}'.format(measure), param_plot_params)

    plot_best(best_plot_data, 'best_{}'.format(measure), best_plot_params)

def plot_param(data, file_name, plot_params):

    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)

    for method in data:
        m_specs = method.split('-')
        ris = m_specs[m_specs.index('RIS') + 1] == 'True'
        if 'n_step' not in m_specs:
            n_step = '(0)'
        else:
            n_step = m_specs[m_specs.index('nstep') + 1]

        if not ris:
            label = 'OIS-TD' + n_step
        else:
            typ = m_specs[m_specs.index('type') + 1]
            k =''
            bufftype = ''
            if typ != 'single' and typ != 'accum':
                k = m_specs[m_specs.index('k') + 1]
                bufftype = m_specs[m_specs.index('bufftype') + 1]

            label = 'RIS' + n_step + typ + k + bufftype
        sorted_lrs = sorted(data[method])
        if len(data[method]) == 0:
            print ('skipping {}'.format(label))
            continue
        x = []
        y = []
        ylower = []
        yupper = []
        for lr in sorted_lrs:
            #print ('number of errors for {} trajs: {}'.format(num_traj, len(data[method][num_traj])))
            err = np.array(data[method][lr])
            y.append(err)
            x.append(lr)

            #x = np.arange(len(y))
        line, = plt.plot(x, y, label=label, marker = 'o')
        color = line.get_color()
        alpha = 0.5

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

def plot_best(data, file_name, plot_params):
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)

    for method in data:
        label = method
        print (label)

        lrs = data[method]
        if len(lrs) != 1:
            continue
        x = []
        y = []
        ylower = []
        yupper = []

        for lr in lrs:
            print ('best lr {}'.format(lr))
            steps = data[method][lr]['x'][0]
            errors = np.array(data[method][lr]['y'])
            
            for idx, st in enumerate(steps):
                errs = errors[:, idx]
                n = len(errs) # number of trials
                mean = np.mean(errs, axis = 0)
                std = np.std(errs, axis = 0)

                yerr = 1.96 * std / np.sqrt(float(n))
                y.append(mean)
                ylower.append(mean - yerr)
                yupper.append(mean + yerr)
                x.append(st)

        #x = np.arange(len(y))
        line, = plt.plot(x, y, label=label)
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

    data = collect_data()
    param_plot_params = {'bfont': 30,
               'lfont': 21,
               'tfont': 25,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': [0 , 2000],
               'x_range': None,
               'log_scale': True,
               'x_label': 'Learning Rate',
               'y_label': 'Mean Squared Error',
               'shade_error': True,
               'x_mult': 1}
    best_plot_params = {'bfont': 30,
               'lfont': 21,
               'tfont': 25,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': None,
               'x_range': None,
               'log_scale': True,
               'x_label': 'Number of Steps',
               'y_label': 'Mean Squared Error',
               'shade_error': True,
               'x_mult': 1}
    lr_plot(data, param_plot_params, best_plot_params, measure = 'perf', learning_curve = True)
    print ('AUC stuff')
    lr_plot(data, param_plot_params, best_plot_params, measure = 'AUC', learning_curve = True)



 
if __name__ == '__main__':
    main()
