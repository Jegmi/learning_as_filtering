#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synaptic Filter: performance simulations.

Created on Wed Apr 24 11:50:39 2019
@author: jannes
"""

# =============================================================================
#  load libs
# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import patches
#matplotlib.rcParams.update({'figure.figsize': (10,6)})
import itertools as it
import pandas as pd
from time import time
from time import sleep
from datetime import datetime
import scipy.stats
import argparse
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal as mnorm
import scipy.special as ss
from scipy.optimize import fmin
from util.util import *
import pprint
matplotlib.rcParams.update({'font.size': 15})
text_size = 13
import os, sys
import pickle

if __name__ == "__main__":
    """ 
    Generating and plotting of figures for "Learning as filtering" manuscript.
    
    To plot, provide one of the following figure keys (str) as -f argument:
        fig1d, fig2a, fig2b, fig2c, fig2d, fig2e, fig3, fig4, 
        figS1, figS2, figS3, figS4, figS5
        
    To generate new figure-data, use the production keys (str) as -p arguments:
        fig1d, fig2_dim, fig2_beta, fig2_dim_pf, fig2_beta_pf, fig2_eta,
        fig2d, fig2e, fig3, fig4, figS4, figS5
        
    Command line example for generating fig1d data and plot:
        python main.py -f fig1d -p fig1d
    
    The plot is saved as ./figures/fig1d.pdf
    The data (a pandas data frame) is stored as ./pkl_data/fig1d/fig1d.pkl
        
    Further details:
        This file contains 2 simulation environments, one for the biological &
        one for the performance oriented simulations. Parameters are set in 
        three layers. Lower layers have priority.
        1. default parameters apply to all simulations
        2. simulation type parameters apply either to bio- or performance sims
        3. for each figure, specpfic parameters can be selected
        
        Plotting parameters (labels, line color ect) must be tuned directly in
        the function "plt_manuscript_figures" in the file "./util/util.py"
    """

    # pars arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule",
                        "-r",
                        default='Exp',
                        help="which rule to run")
    parser.add_argument("--M",
                        "-M",
                        default=1,
                        help="trials to estimate variance",
                        type=int)
    parser.add_argument("--steps",
                        "-T",
                        default=200,
                        help="number of ou times",
                        type=float)
    parser.add_argument("--L",
                        "-L",
                        default=8192,
                        help="number of dim steps",
                        type=int)
    parser.add_argument("--dim",
                        "-d",
                        default=20,
                        help="number of synapses (default: 100)",
                        type=int)
    parser.add_argument("--i",
                        "-i",
                        default=-1,
                        help="run id, -1 means no cluster",
                        type=int)
    parser.add_argument("--f",
                        "-f",
                        default=None,
                        help="figure name as in manuscript")
    parser.add_argument("--p",
                        "-p",
                        default=None,
                        help="if provided, produce new pkl files")
    my_args = parser.parse_args()

    fig = my_args.f
    production = my_args.p
    print('Produce (-p argument)',production,'and plot (-f argument)',fig)

    ## init default pars
    mp, p = {}, {}
    p['beta0'] = 1
    p['tau_s'] = 0.025
    p['tau_s'] = 0.1
    mp['tau_ou'] = 100
    steps = 120
    p['alpha'] = 1
    mp['tau_ss'] = [p['tau_s']]
    mp['alphas'] = [p['alpha']]
    mp['beta0s'] = [p['beta0']]  # 0 - 2
    mp['online'] = True
    mp['bias'] = True
    p['mu_ou'] = 0
    p['sig2_ou'] = 1
    mp['dims-gm'] = [np.nan]
    dim = p['dim'] = mp['dim'] = my_args.dim
    mp['dims'] = [dim]
    # protocol
    mp['STDP_plots'] = False
    mp['hetero-STDP'] = False
    # spikes props
    mp['t_ref'] = 0.01
    mp['ty_ref'] = 0.0
    mp['nu-max'] = 40
    # errorbars
    mp['corr_L'] = 0.01  # length of correlation fct
    mp['corr_downsample'] = 0  # down sampling of corr computation
    mp['M'] = my_args.M
    mp['M-run'] = True  # for paralleli computing on cluster.
    mp['L'] = my_args.L

    # =========================================================================
    #     ## Biological simulations (STDP protocols)
    # =========================================================================
    if (fig in ('fig3', 'figS4', 'fig4',
                'figS5')) or (production in ('fig3', 'figS4', 'fig4',
                                             'figS5')):

        # pars
        dt = 0.00005  # sec
        mp['w-dynamic'] = ['protocol']
        mp['T_wait'] = 0.3  # for relaxation
        mp['T_wait_PC'] = 0.15  # before PC applied
        mp['triplet'] = False
        mp['variable-plots'] = fig in ('figS4', 'figS5')
        mp['dT_plt'] = 0.01  # plot time series at these pre-post +/- intervals
        mp['dB'] = 1  # s, 1/freq burst
        mp['nS'] = 1
        mp['dS'] = 0.05
        mp['nBs'] = [1]
        # time series
        mp['cut_factor'] = 0
        mp['dT_range'] = 0.05

        # init condtions
        mp['mu0_bs'] = [1]
        mp['mu0_1s'] = [1]
        mp['mu0_2s'] = [1]
        mp['Sig0_bs'] = [1]
        mp['Sig0_ws'] = [1]
        mp['Sig0_wws'] = [0]
        mp['Sig0_bws'] = [0]

        mp['biass'] = [1]

        # bias parameters
        mp['tau_oub'] = 0.025
        mp['mu_oub'] = 1
        mp['sig2_oub'] = 1
        # neuron
        mp['g0'] = 1

        # hetero synaptic STDP: w/o bias, correlated or not correlated
        if fig in ('fig4', 'figS5') or (production in ('fig4', 'figS5')):
            mp['rules'] = ['exp', 'corr']
            dt = 0.0001
            mp['M'] = 200
            dt = 0.00001

            mp['w-dynamic'] = ['protocol']
            mp['dims'] = [3]
            dim = 2
            mp['n_PCs'] = [1, 2]
            mp['nBs'] = [1]
            mp['online'] = False
            steps = 0
            mp['dS'], mp['dT'], mp['dT2'] = 0.05, 0.01, 0.01
            mp['nB'] = 1
            mp['alphas'] = np.linspace(0, 1, 11)
            mp['testing-time'] = 0.25  #s
            p['ty_ref'] = 0.1  # [s] 10 Hz # 1/nu_y
            mp['bias'] = False
            mp['hetero-STDP'] = True  # this is for
            mp['hetero-STDP-xSpikes'] = ['homo', 'mixed', 'hetero']
            mp['hetero-STDP-xSpikes'] = ['homo', 'hetero']
            mp['hetero-STDP-xSpikes'] = ['homo']
            mp['nu-max'] = 10
            mp['t_ref'] = 0.0
            mp['tau_ou'] = 10000

            mp['variable-plots'] = 'figS5' in (fig, production)
            if mp['variable-plots'] == True:
                mp['rules'] = ['corr']
                mp['n_PCs'] = [2]
                mp['M'] = 5

        # STDP: w/o bias, correlated or not correlated
        elif fig in ('fig3', 'figS4') or (production in ('fig3', 'figS4')):
            # three conditions:
            # d=1, corr=exp (no bias)
            # d=2, exp (bias)
            # d=2, corr (bias)
            mp['rules'] = ['exp', 'corr']
            mp['w-dynamic'] = ['protocol']
            mp['dims'] = [1, 2]  #
            mp['nBs'] = [1]
            mp['n_PCs'] = [0]
            mp['M'] = 91  # for dT
            mp['online'] = False
            my_args.dim = dim = 1

            mp['variable-plots'] = 'figS4' in (fig, production)
            if mp['variable-plots'] == True:
                mp['rules'] = ['corr']
                mp['n_PCs'] = [2]
                mp['M'] = 5
                mp['dims'] = [2]

        p.update({
            'tau': 0.025,
            'data-duration': int(2 / dt),
            'spikes': True,
            'deterministic': False,
            'lr': np.nan
        })
        mp['ty_ref'] = mp['t_ref'] if 'ty_ref' not in mp else mp['ty_ref']

        dtype = 'TOY'
        p.update({
            dtype: {
                'N': 40
            },
            'ydim': 1,
            'ydim': 1,
            'bN': 4,
            'sN': 50,
            't_test_step': 2,
            'N_sec': 4,
            'factor': 2,
            'MNIST': {
                'path': './../MNIST/'
            },
        })

        # meta pars for sim
        mp['run_bayesian'] = True
        p['gmax'] = 50
        p['g0'] = mp['g0']  # typical
        p['dim'] = p['dim-gm'] = dim
        # set beta with GM (p > 0 otherwise)
        set_beta = lambda p: p['beta0'] * np.log(p['gmax'] / p['g0']) / (
            (p['sig2_ou'] * p['dim-gm'])**0.5 * 4)
        mp['dt'] = dt
        mp.update(p)

        if production is not None:

            plt_path = get_pltpath(my_args_i=my_args.i)
            tab = gen_table(mp)

            # create new data or load?
            sim_ids = np.arange(len(tab))
            for sim_id in sim_ids:
                print('Start sim:', sim_id)
                # update parameters in p with meta parameter table
                mp['g0dt'] = mp['g0'] * dt
                p.update(mp)
                p.update(tab.loc[sim_id])

                t_num = int(np.ceil(steps * p['tau_ou'] / dt))
                p['t_num'] = t_num
                p['bayesian'] = np.isnan(p['lr'])

                p['beta'], tab.loc[sim_id, 'beta'] = 2 * (mp['beta0'], )

                # make int
                p['dim'], p['m'], p['L'] = int(p['dim']), int(p['m']), int(
                    p['L'])
                # vars and init cond.
                tout, t_per_min = 10, 0
                v = Variables()
                v.init_spikes(p)
                tab2v(tab, sim_id, v, p)  # init v

                t = 0
                t0 = time()
                print('Reminder: reduce run time to 0.6')
                while t < (p['t_num'] - 1) * 0.6:
                    k = 0 if mp['online'] else t  # k is array index
                    # load protocol
                    v['x'][k +
                           1] = (1 - dt / p['tau']) * (v['x'][k] + v['Sx'][k])
                    if p['bias']:
                        v['x'][k + 1][0] = 1
                    # likelihood
                    if p['rule'] in ('exp', 'grad', 'exp_sample'):
                        dmu_like, dsig2_like = v.exp(p, k)
                    elif p['rule'] in ('corr', 'corr_sample', 'corrx'):
                        # rm bias correlations
                        if p['rule'] == 'corrx':
                            if p['dim'] == 3:
                                v['sig2'][k][[0, 0, 1, 2], [1, 2, 0, 0]] = 0
                            elif p['dim'] == 2 and p['bias']:
                                v['sig2'][k][[0, 1], [1, 0]] = 0
                        dmu_like, dsig2_like = v.corr(p, k)
                    elif p['rule'] in ('exp_smooth', ):
                        # like with smoothing
                        dmu_like, dsig2_like = v.exp(p, k)
                    elif p['rule'] in ('corr_smooth', ):
                        # like
                        dmu_like, dsig2_like = v.corr(p, k)
                    # prior
                    dmu_pi, dsig2_pi = v.get_prior(p, k)
                    # final update
                    v['mu'][k + 1] = v['mu'][k] + (dmu_like + dmu_pi * dt)
                    v['sig2'][k +
                              1] = v['sig2'][k] + (dsig2_like + dsig2_pi * dt)

                    # error catching
                    if np.any(v['mu'][k + 1] == np.inf) or np.any(
                            np.isnan(v['mu'][k + 1])):
                        print('A weight diverged => break loop', t)
                        break
                    if np.any(np.diag(v['sig2'][k + 1]) < 0):
                        print('Neg variance', np.diag(v['sig2'][k + 1]),
                              '\n => break loop', t)
                        break

                    # t before STDP
                    if t == int(mp['T_wait'] / 2 / dt - 1):
                        tab2v(tab, sim_id, v, p, k=t, lab='i',
                              rev=True)  # before STDP
                    t += 1  # increment
                # end k-loop
                tab2v(tab, sim_id, v, p, k=t - 2, lab='f',
                      rev=True)  # after STDP
                time_min, time_sec = round((time() - t0) / 60, 1), round(
                    (time() - t0), 0)
                unit, time_display = (('min', time_min) if time_min > 2 else
                                      ('s', time_sec))
                print('fin sim_id {1} after {0}{2}:'.format(
                    time_display, sim_id, unit))
                # report failures:
                for key in ['w<0', 'gdt>1', 'gbardt>1']:
                    tab.loc[sim_id, key] = p[key]

                if mp['variable-plots']:  # plot
                    dT_match = (np.round(np.abs(p['dT']), 4) == p['dT_plt'])
                    ##STDP negative lobe time series
                    if ((fig == 'figS4') and dT_match):
                        dt = p['dt']
                        i0, i1 = int(0.3 / dt), int(0.4 / dt)
                        figS4 = {
                            'tspan': np.linspace(i0 * dt, (i1 - 1) * dt,
                                                 i1 - i0),
                            'bias': v['mu'][i0:i1, 0],
                            'mu1': v['mu'][i0:i1, 1],
                            'cov0': v['sig2'][i0:i1, 0, 0],
                            'cov1': v['sig2'][i0:i1, 1, 1],
                            'cov12': v['sig2'][i0:i1, 0, 1],  # w-b cov
                            'x1': v['x'][i0:i1, 1],
                            'y0': v['y'][i0:i1],  # spike train
                            'dT': p['dT']
                        }
                        suf = 'prepost' if p['dT'] > 0 else 'postpre'
                        save_obj(figS4, 'figS4_' + suf, './pkl_data/figS4/')

                    # HETERO positive lobe time series
                    if ((fig == 'figS5') and dT_match
                            and p['hetero-correlations']):
                        dt = p['dt']
                        i0, i1 = int(0.3 / dt), int(0.4 / dt)
                        figS5 = {
                            'tspan': np.linspace(i0 * dt, (i1 - 1) * dt,
                                                 i1 - i0),
                            'bias': v['mu'][i0:i1, 0],
                            'mu1': v['mu'][i0:i1, 1],
                            'mu2': v['mu'][i0:i1, 2],
                            'cov0': v['sig2'][i0:i1, 0, 0],
                            'cov1': v['sig2'][i0:i1, 1, 1],
                            'cov12': v['sig2'][i0:i1, 1, 2],  # w-w cov
                            'x1': v['x'][i0:i1, 1],
                            'x2': v['x'][i0:i1, 2],
                            'y0': v['y'][i0:i1],  # spike train
                            'dT': p['dT']
                        }
                        suf = 'prepost' if p['dT'] > 0 else 'postpre'
                        save_obj(figS5, 'figS5_' + suf, './pkl_data/figS5/')

            # for non-time series plots, end learning rate loop
            if mp['variable-plots'] == False:
                print('final report. Rule:', p['rule'], 'sim_id:', sim_id)
                save_obj(tab, production, './pkl_data/{0}/'.format(production))

# =============================================================================
#     ## computational performance simulations
# =============================================================================
    else:  # # fig1, fig2, figS1, figS2, figS3

        dt = 0.0005
        mp['lrs'] = [np.nan]
        mp['w-dynamic'] = ['OU']
        mp['T_wait'] = 1  #s
        mp['mets'] = [
            'MSE', 'p_in', 'z', 'z2', 'z2d', 'z2d_pt', 'z2d0', 'z2dg'
        ]
        mp['cut_factor'] = 1

        if 'fig1d' in (fig, production):  # time series plot
            folder = production
            mp['rules'] = ['corr']
            mp['cut_factor'] = 1
            mp['M'] = 1
            dt = 0.001
            mp['tau_ou'] = 100
            mp['sig2_ou'] = 1
            #        mp['online'] = False
            mp['dims-gm'] = [2]
            mp['lrs'] = expspace(0.01, 0.1, 3)
            mp['dims'] = np.array([2], dtype=np.int8)  # other dims above
            steps = 6
            mp['online'] = False
            mp['corr_downsample'] = 0  # no var
            mp['downsample'] = 100  # time series plotting
            mp['corr_L'] = 0.1  # length of correlation fct
            mp['ty_ref'] = 0.0
            np.random.seed(4)

        # figure 2: MSE, z-plots
        elif (fig in ('fig2b', 'figS2b', 'figS1b')
              or (production == 'fig2_beta')):
            folder = production  # 'fig2abc'
            mp['rules'] = ['exp', 'corr', 'exp_smooth', 'corr_smooth']
            mp['beta0s'] = [0, 0.33, 0.67, 1, 1.33, 1.67, 2]
            mp['dims'] = np.array([5, 15], dtype=np.int8)
            mp['tau_ou'] = 100
            mp['M'] = 100
            mp['M-run'] = False  # true
            steps = 11

        # MSE, z-plots
        elif (fig in ('fig2a', 'figS2a', 'figS1a')
              or (production == 'fig2_dim')):
            folder = production  #'fig2abc'
            mp['rules'] = ['exp', 'corr', 'exp_smooth', 'corr_smooth']
            mp['beta0s'] = [0.33, 1]
            mp['dims'] = np.array([1, 2, 7, 10, 12], dtype=np.int8)
            mp['tau_ou'] = 100
            mp['M'] = 100
            mp['M-run'] = False  # true
            steps = 11

        elif production == 'fig2_dim_pf':
            folder = production
            mp['rules'] = ['pf_corr']
            mp['beta0s'] = [1]
            dt = 0.001
            mp['dims'] = np.array([1, 2, 5, 7, 10, 12, 15], dtype=np.int8)
            mp['tau_ou'] = 100
            mp['M'] = 100
            mp['M-run'] = False
            steps = 10

        elif production == 'fig2_beta_pf':
            folder = production
            mp['rules'] = ['pf_corr']
            mp['beta0s'] = [0, 0.33, 0.67, 1.33, 1.67, 2]
            dt = 0.001
            mp['dims'] = np.array([5], dtype=np.int8)
            mp['tau_ou'] = 100
            mp['M'] = 100
            mp['M-run'] = False
            steps = 10

        elif ((fig == 'fig2c') or (production == 'fig2_eta')):
            folder = production
            mp['rules'] = ['grad']
            mp['lrs'] = expspace(0.001, 3, 21)
            mp['beta0s'] = [0.33, 1]
            mp['dims'] = np.array([5, 15], dtype=np.int8)
            mp['tau_ou'] = 100
            # new blog
            mp['M'] = 100
            mp['M-run'] = False  # true
            steps = 11

        elif 'fig2d' in (fig, production):
            folder = production
            mp['rules'] = ['corr', 'exp']
            mp['dims'] = np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20
            ],
                                  dtype=np.int8)
            mp['dims'] = np.array([1], dtype=np.int8)
            mp['tau_ou'] = 5
            mp['lrs'] = expspace(0.05, 2, 11)
            mp['M'] = 100
            mp['M-run'] = False  # true
            steps = 300

        elif 'fig2e' in (fig, production):
            folder = production
            mp['rules'] = ['grad', 'corr', 'exp']
            mp['tau_ou'] = 5
            mp['cut_factor'] = 1
            mp['dims'] = np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20
            ],
                                  dtype=np.int8)
            mp['dims'] = np.array([8], dtype=np.int8)
            mp['dims-gm'] = [5]
            mp['lrs'] = expspace(0.05, 2, 11)
            mp['M'] = 100
            mp['M-run'] = True
            steps = 300

        mp['steps'] = steps
        p.update({
            'tau': 0.025,
            'data-duration': int(2 / dt),
            'spikes': True,
            'deterministic': False,
            'lr': np.nan
        })
        mp['ty_ref'] = mp['t_ref'] if 'ty_ref' not in mp else mp['ty_ref']

        dtype = 'TOY'
        p.update({
            dtype: {
                'N': 40
            },
            'ydim': 1,
            'ydim': 1,
            'bN': 4,
            'sN': 50,
            't_test_step': 2,
            'N_sec': 4,
            'factor': 2,
            'MNIST': {
                'path': './../MNIST/'
            },
        })

        # meta pars for sim
        mp['run_bayesian'] = True
        p['gmax'] = 50
        p['g0'] = 1
        p['dim'] = p['dim-gm'] = dim
        # set beta with GM (p > 0 otherwise)
        set_beta = lambda p: p['beta0'] * np.log(p['gmax'] / p['g0']) / (
            (p['sig2_ou'] * p['dim-gm'])**0.5 * 4)
        mp['dt'] = dt
        p['beta'] = set_beta(p)
        p['g0dt'] = p['g0'] * dt
        # different init values
        p['init_deviation'] = (1 + eta(dim) * (mp['hetero-STDP'] == False))
        mp.update(p)
        mp['plot-series'] = False
        mp['plot-MSE'] = False
        mp['syns-to-plot'] = np.array([0, 10, 20, 30, 50,
                                       100 - 1])  # 1,2,3,10,20,50,60,90)
        mp['syns-to-plot'] = mp['syns-to-plot'][mp['syns-to-plot'] < dim]

        # produce new pkl files
        if production is not None:
            tab = gen_table(mp)
            if my_args.i > -1:  # on cluster
                sim_ids = tab[tab.m == np.arange(mp['M'])[
                    my_args.i]].index if mp['M-run'] else [
                        np.arange(len(tab))[my_args.i]
                    ]
                print('sim_ids', sim_ids)
            else:  # on local machine
                sim_ids = np.arange(len(tab))

            for sim_id in sim_ids:
                print('Start sim:', sim_id)
                # update parameters in p with meta parameter table: same name!
                p.update(mp)
                p.update(tab.loc[sim_id])

                t_num = int(np.ceil(steps * p['tau_ou'] / dt))
                p['t_num'] = t_num
                p['k_cut'] = int(p['cut_factor'] * p['tau_ou'] / dt)
                p['k_cut'] = p['k_cut'] if t_num > p['k_cut'] else 0
                p['dW'] = (dt * p['sig2_ou'] / p['tau_ou'] * 2)**0.5
                p['bayesian'] = np.isnan(p['lr'])
                p['beta'], tab.loc[sim_id, 'beta'] = 2 * (
                    2 if production == 'fig1d' else set_beta(p), )

                # make int
                p['dim'], p['m'], p['L'] = int(p['dim']), int(p['m']), int(
                    p['L'])
                if my_args.i > -1:  # on cluster
                    if sim_id == sim_ids[0]:  # only first
                        print('')
                        pprint.pprint(p)
                        print('')

                # vars and init cond.
                tout, t_per_min, err = 10, 0, np.zeros(len(p['mets']))
                v = Variables()

                # init online mean and sem computation
                tsac = TimeSeries_AutoCorrelation(
                    dim=len(p['mets']),
                    L=int(p['corr_L'] * p['tau_ou'] / p['dt']),
                    downsample=p['corr_downsample'])

                if p['w-dynamic'] == 'protocol':
                    v.init_spikes(p)
                else:
                    v.init(2 if mp['online'] else p['t_num'], p)
                p['nu*dt'] = p['nu-max'] * dt * np.ones(p['dim'])
                if p['dim-gm'] > p['dim']:
                    p['nu*dt-gm'] = p['nu*dt'][0] * np.ones(p['dim-gm'] -
                                                            p['dim'])

                t = 0
                t0 = time()

                while t < p['t_num'] - 1:
                    #        for t in np.arange(0,p['t_num']-1): # make sure things dont explode
                    k = 0 if mp['online'] else t  # k is array index
                    # update the world
                    v.run_world(p, t, k)

                    # likelihood
                    if p['rule'] in ('exp', 'grad', 'exp_sample'):
                        dmu_like, dsig2_like = v.exp(p, k)
                        dsig2_pi = -2 * (
                            v['sig2'][k] -
                            p['sig2_ou']) / p['tau_ou'] if p['bayesian'] else 0
                        dmu_pi = -(v['mu'][k] - p['mu_ou']
                                   ) / p['tau_ou'] if p['bayesian'] else 0
                        # final update
                        v['mu'][k + 1] = v['mu'][k] + (dmu_like + dmu_pi * dt)
                        v['sig2'][k + 1] = v['sig2'][k] + (dsig2_like +
                                                           dsig2_pi * dt)

                    elif p['rule'] in ('corr', 'corr_sample'):
                        dmu_like, dsig2_like = v.corr(p, k)
                        dsig2_pi = -2 * (v['sig2'][k] - np.diag(
                            np.ones(p['dim']) * p['sig2_ou'])) / p['tau_ou']
                        dmu_pi = -(v['mu'][k] - p['mu_ou']) / p['tau_ou']
                        # final update
                        v['mu'][k + 1] = v['mu'][k] + (dmu_like + dmu_pi * dt)
                        v['sig2'][k + 1] = v['sig2'][k] + (dsig2_like +
                                                           dsig2_pi * dt)

                    elif p['rule'] in ('exp_smooth', ):
                        # like with smoothing
                        dmu_like, dsig2_like = v.exp(p, k)
                        # prior
                        dsig2_pi = -2 * (
                            v['sig2'][k] -
                            p['sig2_ou']) / p['tau_ou'] if p['bayesian'] else 0
                        dmu_pi = -(v['mu'][k] - p['mu_ou']
                                   ) / p['tau_ou'] if p['bayesian'] else 0
                        v['mu'][k + 1] = v['mu'][k] + (dmu_like + dmu_pi * dt)
                        v['sig2'][k + 1] = v['sig2'][k] + (dsig2_like +
                                                           dsig2_pi * dt)

                    elif p['rule'] in ('corr_smooth', ):
                        # like
                        dmu_like, dsig2_like = v.corr(p, k)
                        # prior
                        dsig2_pi = -2 * (v['sig2'][k] - np.diag(
                            np.ones(p['dim']) * p['sig2_ou'])) / p['tau_ou']
                        dmu_pi = -(v['mu'][k] - p['mu_ou']) / p['tau_ou']
                        v['mu'][k + 1] = v['mu'][k] + (dmu_like + dmu_pi * dt)
                        v['sig2'][k + 1] = v['sig2'][k] + (dsig2_like +
                                                           dsig2_pi * dt)

                    if np.any(v['mu'][k + 1] == np.inf) or np.any(
                            np.isnan(v['mu'][k + 1])):
                        print('A weight diverged => break loop', t)
                        break  #[]**2

                    ## particle filter
                    if 'pf' in p['rule']:
                        v.pf(p, k)  # update everything internally

                    # increment
                    if p['online']:
                        # update
                        # shift all variables back
                        v['gbar'][k + 1] = v['gbar'][k]
                        v['g'][k + 1] = v['g'][k]
                        v['y'][k + 1] = v['y'][k]
                        v['gmap'][k + 1] = v['gmap'][k]
                        for key in v.keys():
                            if key not in ('t', 'y_ref', 'x_ref', 't-gm',
                                           'x_ref-gm'):
                                v[key][0] = v[key][1]

                        # compute error for online estimate
                        #p['k_cut'] = 0
                        #tsac.count = 0
                        if t >= p['k_cut']:
                            if t == p['k_cut']:
                                t1 = round(time() - t0, 2)
                                print('Finished burn in', t1, '[s]')
                                t_print = 60

                            if time() - t0 > t1 + t_print:
                                steps_per_min = 60 * (t - p['k_cut']) / (
                                    time() - t0 - t1)
                                min_done = round((time() - t0) / 60, 1)
                                min_togo = round(
                                    (p['t_num'] - t) / steps_per_min, 1)
                                print('Step:', t, 'of', p['t_num'], '\n',
                                      'Done:', min_done, 'min \n', 'ToGo:',
                                      min_togo, 'min \n', 'Total:',
                                      round((min_done + min_togo) / 60), 'h')
                                t_print += 300  # next printing in 300s
                                #tsac.count = 0

                            state = v.res(p, k=0, end=-1)
                            tsac.run_online(state)

                    elif 'pf' in p['rule']:  # shift particle filter always
                        v['p'][0] = v['p'][1]

                    # increment
                    t += 1
                # end k-loop
                time_min, time_sec = round((time() - t0) / 60, 1), round(
                    (time() - t0), 0)
                unit, time_display = (('min', time_min) if time_min > 2 else
                                      ('s', time_sec))
                print('fin sim_id {1} after {0}{2}:'.format(
                    time_display, sim_id, unit))
                if mp['online'] == False:
                    if production == 'fig1d':  # save down_sampled time series

                        n_sig = 2
                        k_cut = p['k_cut']
                        dt = p['dt']
                        dim = 1
                        d = p['downsample']
                        t_num = p['t_num']
                        m, s2 = ('mu', 'sig2')
                        tspan = (np.arange(0, t_num - 1 - k_cut) * dt)[::d]
                        # ground truth
                        yplt_gt = v['lam'][k_cut:-1:d, dim]
                        # filter
                        yplt = v[m][k_cut:-1:d, dim]
                        if len(v[s2][k_cut:-1:d].shape) > 1:
                            err = n_sig * v[s2][k_cut:-1:d, dim, dim]**0.5
                        else:
                            err = n_sig * v[s2][k_cut:-1:d, dim]**0.5
                        fig1d = {
                            'tspan': tspan,
                            'yplt_gt': yplt_gt,
                            'yplt': yplt,
                            'err': err
                        }
                        save_obj(fig1d, production,
                                 './pkl_data/{0}/'.format(folder))

                    else:
                        # only mean
                        tab.loc[sim_id, p['mets']] = v.res(p,
                                                           k=p['k_cut'],
                                                           end=-1)
                else:
                    # mean (and sem)
                    tab.loc[sim_id, p['mets']] = tsac.post_process(
                        include_sem=False)

                # report failures:
                for key in ['w<0', 'gdt>1', 'gbardt>1']:
                    tab.loc[sim_id, key] = p[key]
                # cluster normal run
                if my_args.i != -1 and mp['M-run'] == False:
                    save_obj(tab, '{1}_simtab{0}'.format(sim_id, production),
                             './pkl_data/{2}/'.format(folder))
                # cluster m-run
                if my_args.i != -1 and mp['M-run'] == True:
                    save_obj(tab, '{1}_m{0}'.format(p['m'], production),
                             './pkl_data/{2}/'.format(folder))

                # end learning rate loop
                print('---')
                print('final report. Rule:', p['rule'], 'sim_id:', sim_id)
                print('')
                print(tab[np.isnan(tab.beta) == False].T)
                print('---')

# =============================================================================
#     ## Plotting figures
# =============================================================================

    if fig is not None:
        p.update(mp)  # in case plotting relies on p

        # plotting
        plt_manuscript_figures(fig, mp)
        print('Plotted',fig,'in ./figures')
