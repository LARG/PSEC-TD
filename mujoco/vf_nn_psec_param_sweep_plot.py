"""Plot value function results."""
import argparse
from protos import results_pb2
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
from matplotlib import pyplot as plt

from matplotlib import rcParams
import os

from scipy import stats


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

    for method in sorted(data):
        label = method
        print ('method ' + method)

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

        #x = np.arange(len(y))
        linestyle = '-'
        if method == 'TD(0)':
            line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 5, color = 'red')
        else:    
            line, = plt.plot(x, y, label=label, linestyle = linestyle, linewidth = 5)
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

    for basename in os.listdir(FLAGS.result_directory):
        filename = os.path.join(FLAGS.result_directory, basename)
        if '.err' in filename or '.out' in filename or '.log' in filename or '.nfs' in filename:
            continue
        try:
            results = read_proto(filename)
        except Exception as e:
            if 'Tag' not in str(e):
                raise e

        print (basename)
        #method_TD(0)_trial_847277_trajs_9_alpha_0.010000_tiling_20_t1_15_t2_9_psec-act_0_psec-nh_0_psec-ne_0_psec-lr_0.000000
        #method_TD(0)_trial_847277_trajs_9_alpha_0.010000_tiling_20_t1_15_t2_9_psec-act_0_psec-nh_0_psec-ne_0_psec-lr_0.000000
        #print (basename)
        specs = basename.split('_')
       
        # thing that is fixed is the seed and traj size, finding best performing based on these two fixed points
        # if td looking at lr, tc, etc
        # if psec, looking at above AND the extra PSEC stuff
 
        method = specs[specs.index('method') + 1]
        if 'Finetune' in method:
            continue
        lr = specs[specs.index('alpha') + 1]
        tiling = specs[specs.index('vf-act') + 1]
        tile_1 = specs[specs.index('vf-nh') + 1]
        tile_2 = specs[specs.index('vf-ne') + 1]
        
        num_trajs = int(results.num_trajs) #int(results.num_steps_observed[0])

        if method not in data:
            data[method] = {}       
        if num_trajs not in data[method]:
            data[method][num_trajs] = {}

        if lr not in data[method][num_trajs]:
            # list of final errors across all trials for a given batch size and lr
            data[method][num_trajs][lr] = {}

        if tiling not in data[method][num_trajs][lr]:
            data[method][num_trajs][lr][tiling] = {}

        if tile_1 not in data[method][num_trajs][lr][tiling]:
            data[method][num_trajs][lr][tiling][tile_1] = {}
        if tile_2 not in data[method][num_trajs][lr][tiling][tile_1]:
            data[method][num_trajs][lr][tiling][tile_1][tile_2] = {}

        psec_act = specs[specs.index('psec-act') + 1]
        psec_nh = specs[specs.index('psec-nh') + 1]
        psec_ne = specs[specs.index('psec-ne') + 1]
        psec_lr = specs[specs.index('psec-lr') + 1]

        if psec_act not in data[method][num_trajs][lr][tiling][tile_1][tile_2]:
            data[method][num_trajs][lr][tiling][tile_1][tile_2][psec_act] = {}
        
        if psec_nh not in data[method][num_trajs][lr][tiling][tile_1][tile_2][psec_act]:
            data[method][num_trajs][lr][tiling][tile_1][tile_2][psec_act][psec_nh] = {}
         
        if psec_ne not in data[method][num_trajs][lr][tiling][tile_1][tile_2][psec_act][psec_nh]:
            data[method][num_trajs][lr][tiling][tile_1][tile_2][psec_act][psec_nh][psec_ne] = {}       

        if psec_lr not in data[method][num_trajs][lr][tiling][tile_1][tile_2][psec_act][psec_nh][psec_ne]:
            data[method][num_trajs][lr][tiling][tile_1][tile_2][psec_act][psec_nh][psec_ne][psec_lr] = []  

        errs = results.value_error
        data[method][num_trajs][lr][tiling][tile_1][tile_2][psec_act][psec_nh][psec_ne][psec_lr].append(errs)
        #data[method][num_trajs][lr][tiling][tile_1][tile_2].append(errs)
        #data[td_method][num_trajs][lr].append(results.td_value_error)
        #data[ristd_method][num_trajs][lr].append(results.ris_td_value_error)

    print ('------- {}'.format(c))
    return data


def best_lr(data):

    min_data = None
    best_data = {}

    err1 = []
    err2 = []


    meth_stats = {}

    for method in data:

        best_data[method] = {}
        for num_traj in data[method]:
            
            min_err = float('inf')
            min_err_lr = -1
            best_spec = ''
            for lr in data[method][num_traj]:
                for ti in data[method][num_traj][lr]:
                    for t1 in data[method][num_traj][lr][ti]:
                        for t2 in data[method][num_traj][lr][ti][t1]:
                            for psec_act in data[method][num_traj][lr][ti][t1][t2]:
                                for psec_nh in data[method][num_traj][lr][ti][t1][t2][psec_act]:
                                    for psec_ne in data[method][num_traj][lr][ti][t1][t2][psec_act][psec_nh]:
                                        for psec_lr in data[method][num_traj][lr][ti][t1][t2][psec_act][psec_nh][psec_ne]:                           
                                            errs = np.array(data[method][num_traj][lr][ti][t1][t2][psec_act][psec_nh][psec_ne][psec_lr])
                                            mean = np.mean(errs)
                                            if psec_nh == '3':
                                                #print (", ".join(map(str, errs)))
                                                if 'PSEC' in method:
                                                    err1 = errs
                                            if method == 'TD(0)':
                                                err2 = errs
                                            spec_state = 'method {}, lr: {} vf-act: {} vf-nh: {} vf-ne: {}, psec act {}, nh {}, ne {}, lr {}, err mean {:.3f}, 95 stderr {:.3f}, reg std {:.3f}, trials: {}'\
                                                    .format(method, lr, ti, t1, t2, psec_act, psec_nh, psec_ne, psec_lr, mean, 1.96 * np.std(errs) / np.sqrt(float(len(errs))), np.std(errs), len(errs))

                                            print ('stats {}'.format(spec_state))
                                            if mean < min_err:
                                                meth_stats[method + '_' + psec_nh] = errs
                                                min_err = mean
                                                min_data = errs
                                                best_spec = 'best method {}, lr: {} vf-act: {} vf-nh: {} vf-ne: {}, psec act {}, nh {}, ne {}, lr {}, err {}, trials: {}'\
                                                    .format(method, lr, ti, t1, t2, psec_act, psec_nh, psec_ne, psec_lr, min_err, len(errs))
            best_data[method][num_traj] = min_data
            print ('traj {} and spec {}'.format(num_traj, best_spec))



    #td_stats = meth_stats['TD(0)']
    for m in meth_stats:
        if 'PSEC' in m:
            sp = m.split('_')[-1]
            print (np.mean(meth_stats['TD(0)_0']))
            print (np.mean(meth_stats[m]))
            print('other method {} and res {}'.format(m, stats.ttest_ind(meth_stats['TD(0)_0'],  meth_stats[m], equal_var = False)))

    c = 0
    '''
    for p, t in zip(err1, err2):
        if p < t:
            c += 1

    print (c / len(err1))

    print ('psec')
    print (err1)

    print ('td')
    print (err2)
    '''

    return best_data 

def main():

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return

    data = collect_data() 


    #print (data)
    best_lr_data = best_lr(data)

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
               'y_label': 'MSVE',
               'shade_error': True,
               'x_mult': 1}
    plot_data(best_lr_data, 'temp1', plot_params)
    
if __name__ == '__main__':
    main()
