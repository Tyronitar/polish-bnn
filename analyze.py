import os
import sys

import argparse
import numpy as np
import glob 
import matplotlib.pylab as plt

from model import resolve_single, normalize_bnn, resolve_bnn, resolve_float
from model.common import bnn_output, psnr_float, resolve_t, ssim_float
from utils import load_image, plot_sample
from model.wdsr import wdsr_b
from model.bnn import wdsr_bnn
import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def precision_recall(gt, mu, u_a, u_e):
    ps = np.linspace(0, 100)
    precision_a = np.zeros(ps.shape)
    precision_e = np.zeros(ps.shape)
    sq_diff = (gt - mu)**2

    percentiles_a = np.percentile(u_a, ps)
    for i, p in enumerate(percentiles_a):
        precision_a[i] = np.sqrt(np.mean(sq_diff[u_a <= p]))

    percentiles_e = np.percentile(u_e, ps)
    for i, p in enumerate(percentiles_e):
        precision_e[i] = np.sqrt(np.mean(sq_diff[u_e <= p]))
    
    return ps, precision_a, precision_e


def calibration(gt, mu, u_a, u_e):
    ps = np.linspace(0, 100) / 100
    freq_a = np.ones(ps.shape)
    freq_e = np.ones(ps.shape)
    resid = np.abs(gt - mu)

    # Scale factor b for Laplace distribution. Var = 2b^2
    b_a = 0.5 * np.sqrt(u_a)
    b_e = 0.5 * np.sqrt(u_e)
    b = 0.5 * np.sqrt(u_a + u_e)

    for i, p in enumerate(ps):
        if p == 1:
            continue
        # Threshold t. p = Pr[mu - t <= x <= mu + t] = Pr[|mu - x| <= t]
        t_a = -b_a * np.log(1 - p)
        t_e = -b_e * np.log(1 - p)
        freq_a[i] = np.count_nonzero(resid <= t_a) / resid.size
        freq_e[i] = np.count_nonzero(resid <= t_e) / resid.size
    
    return ps, freq_a, freq_e


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog="analyze.py",
                                   description="Convert npy to fits mainly.")
    parser.add_argument('indir', type=str, help='In directory')
    parser.add_argument('fn_model', type=str, help='Model .h5 filename')
    parser.add_argument('-r', '--scale', dest='scale', type=int,
                      help="Upsample factor", default=4)
    parser.add_argument('-T', '--num_samples', type=int,
                      help="Number of Monte Carlo samples", default=20)
    parser.add_argument('-b', '--bnn', action='store_true',
                      help="Whether the model is a BNN")

    args = parser.parse_args()
    indir = args.indir
    fn_model = args.fn_model

    if args.bnn:
        model = wdsr_bnn(scale=args.scale, num_res_blocks=32)
    else:
        model = wdsr_b(scale=args.scale, num_res_blocks=32)

    model.load_weights(fn_model)

    fdiroutPLOT= indir+'/plot/'
    fdirinHR = indir+'/POLISH_valid_HR/'
    fdirinLR = indir+'/POLISH_valid_LR_bicubic/X%d/'%args.scale

    if not os.path.isdir(fdiroutPLOT):
        print("Making output plot directory")
        os.system('mkdir -p %s' % fdiroutPLOT)
    
    # List of HR validation files
    # flvalid = glob.glob(indir+'/POLISH_valid*/X%d/*.npy'%args.scale)
    flvalid = glob.glob(fdirinHR + '/*.npy')
    N = len(flvalid)

    precision = np.zeros((N, 2, 50))
    freq = np.zeros((N, 2, 50))
    PSNR = np.zeros(N)
    SSIM = np.zeros(N)
    
    for i, fn_HR in enumerate(flvalid):
        # Get corresponding LR file
        fn_LR = fdirinLR + fn_HR.split('/')[-1][:-4]+'x%d.npy'%args.scale

        # Load files
        datahr = np.load(fn_HR)
        datalr = np.load(fn_LR)
        datalr = datalr[None,:,:]
        
        if args.bnn:
            # Get mean and uncertainty predictions
            mu, au, eu = bnn_output(model, datalr, T=args.num_samples)
        else:
            mu = resolve_float(model, datalr).numpy().squeeze()

        # Calculate PSNR and SSIM
        PSNR[i] = psnr_float(datahr, mu[..., None])
        SSIM[i] = ssim_float(tf.convert_to_tensor(datahr, dtype=tf.float32), tf.convert_to_tensor(mu[..., None]))

        if args.bnn:
            # Calculate calibration
            _, freq_a, freq_e = calibration(datahr.squeeze(), mu, au, eu)
            freq[i, 0, ...] = freq_a
            freq[i, 1, ...] = freq_e

            # Calculate precision_recall
            _, pre_a, pre_e = precision_recall(datahr.squeeze(), mu, au, eu)
            precision[i, 0, ...] = pre_a 
            precision[i, 1, ...] = pre_e

    print(f'Mean PSNR: {np.mean(PSNR)}')
    print(f'Mean SSIM: {np.mean(SSIM)}')

    if args.bnn:
        # plot precision vs recall
        ps = np.linspace(0, 100) / 100

        avg_precision = np.mean(precision, axis=0)
        # print(avg_precision)

        plt.figure(figsize=(8,5))
        plt.plot(ps, avg_precision[0], 'r', label='Aleatoric')
        plt.plot(ps, avg_precision[1], 'b--', label='Epistemic')
        plt.legend()
        plt.xlabel('Recall')
        plt.ylabel('Precision (RMSE)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fdiroutPLOT + 'precision_recall.png')
        plt.close()

        avg_freq = np.mean(freq, axis=0)
        # print(avg_freq)

        plt.figure(figsize=(8,5))
        plt.plot(ps, avg_freq[0], 'r', label='Aleatoric')
        plt.plot(ps, avg_freq[1], 'b--', label='Epistemic')
        plt.plot([0,1],[0,1], 'k')
        plt.margins(x=0, y=0)
        plt.legend()
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(fdiroutPLOT + 'calibration.png')
        plt.close()
