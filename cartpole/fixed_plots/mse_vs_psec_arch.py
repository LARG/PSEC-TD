#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import gym
import os
import sys, time, argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess as sp

nice_fonts = {
        "pgf.texsystem": "pdflatex",
        # Use LaTex to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 27,
        "font.size": 20,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 20,
        "xtick.labelsize": 27,
        "ytick.labelsize": 27,
}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot(x, y, weight, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.plot(x, y, label = r'$\alpha$ = ' + str(weight))

def main():
    plt.figure()
    plt.style.use('seaborn')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams.update(nice_fonts)
    sns.set(rc = nice_fonts)
    #plt.rc('font', **nice_fonts)
    #current_palette = sns.color_palette('Set2')
    #current_palette = flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    #colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    #current_palette = sns.xkcd_palette(colors)
    #current_palette = sns.color_palette("Blues")
    #sns.palplot(current_palette)
    
    data = np.array([
                [81.278], 
                [69.082], 
                [62.048], 
                [60.216], 
                [101.89], 
                ])
    
    data_std = np.array([
                [13.183], 
                [12.008], 
                [10.863], 
                [11.976], 
                [17.875], 
                ])
    
    x_labels = ['0-0', '1-16', '2-16', '3-16', 'TD(0)']
    length = len(data)

    # Set plot parameters
    fig, ax = plt.subplots(figsize = (10,5))#(6, 5))
    width = 0.25 # width of bar
    x = np.arange(length)

    '''
    ax.bar(x, data[:,0], width, label='GAIL*', yerr=data_std[:,0], color = 'tab:blue', edgecolor = 'None')
    ax.bar(x + (1 *  width), data[:,1], width, label='GAIL* + RL', yerr=data_std[:,1], color = 'tab:purple', edgecolor = 'None')
    ax.bar(x + (2 * width), data[:,2], width, label='GAIfO', yerr=data_std[:,2], color = 'tab:pink', edgecolor = 'None')
    ax.bar(x + (3 * width), data[:,3], width, label='GAIfO + RL', yerr=data_std[:,3], color = 'tab:olive', edgecolor = 'None')
    ax.bar(x + (4 * width), data[:,4], width, label='BCO', yerr=data_std[:,4], color = 'tab:orange', edgecolor = 'None')
    ax.bar(x + (5 * width), data[:,5], width, label='TRPO/PPO', yerr=data_std[:,5], color = 'tab:cyan', edgecolor = 'None')
    ax.bar(x + (6 * width), data[:,6], width, label='RIDM (ours)**', yerr=data_std[:,6], color = 'tab:red', edgecolor = 'None')
    '''
    
    #ax.bar(x + (0 * width), data[:,0], width, label='TD(0)', yerr=data_std[:,0], color = 'tab:blue', edgecolor = 'None')
    ax.bar(x[0] + (0 * width), data[0,0], width,yerr=data_std[0,0], color = 'tab:blue', edgecolor = 'None', alpha = 0.5)
    ax.bar(x[1:4] + (0 * width), data[1:4,0], width,yerr=data_std[1:4,0], color = 'tab:blue', edgecolor = 'None')
    ax.bar(x[4:] + (0 * width), data[4:,0], width,yerr=data_std[4:,0], color = 'tab:red', edgecolor = 'None')
    #ax.bar(x + (1 * width), data[:,1], width, label='PSEC-TD(0)', yerr=data_std[:,1], color = 'tab:red', edgecolor = 'None')
    

    ax.set_ylabel('MSVE')
    #ax.set_ylim(0,1.15)
    ax.set_xticks(x + width/2)#+ width/2)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('PSEC Model Architecture')
    #ax.set_title('Comparison of RIDM (ours) Against Established Baseline')
    ax.legend()
    #plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    fig.tight_layout()
    #plt.tight_layout()
    #plt.savefig('/projects/agents2/villasim/brahmasp//temp.pdf')
    plt.savefig('temp1.pdf')
    plt.close()

if __name__ == '__main__':
    main()

