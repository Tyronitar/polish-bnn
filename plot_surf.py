import os
import argparse
import glob 

import optparse
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from PIL import Image

from model import resolve_single, normalize_bnn, resolve_bnn, resolve_float
from model.common import bnn_output, psnr_float, resolve_t, ssim_float
from utils import load_image, plot_sample
from model.wdsr import wdsr_b
from model.bnn import wdsr_bnn
import tensorflow as tf

plt.rcParams.update({
                    'font.size': 8,
                    'font.family': 'serif',
                    'axes.labelsize': 14,
                    'axes.titlesize': 18,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    # 'lines.linewidth': 0.5,
                    'lines.markersize': 5,
                    'legend.fontsize': 14,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})


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


def plot_reconstruction(datalr, datasr, datahr=None, vm=1, 
                        nsub=2, cmap='afmhot', gamma=1/2.2):
    """ Plot the dirty image, POLISH reconstruction, 
    and (optionally) the high resolution true sky image
    """
    nbit = 16
    vminlr=0
    vmaxlr=22500
    vminsr=0
    vmaxsr=22500
    vminhr=0
    vmaxhr=22500
    vmaxhr=2**16-1

    if nsub==5:
        fig = plt.figure(figsize=(16,6))
    if nsub==6:
        fig = plt.figure(figsize=(19,6))

    ax1 = plt.subplot(1,nsub,1)
    plt.title('(a) True sky', c='k', fontsize=17)
    hr = datahr
    print(f'hr range: [{np.min(datahr)}, {np.max(datahr)}]')
    im1 = plt.imshow(hr, cmap=cmap, vmax=1, vmin=-1, 
            aspect='auto', extent=[0,1,0,1])
    plt.axis('off')
    # print(tf.image.psnr(datahr, datasr, max_val=2.0)[0])
    # print(20 * np.log10(2.0) - 10 * np.log10(np.mean((datahr - datasr)**2)))
    plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, fraction=0.05)

    ax2 = plt.subplot(1,nsub,2, sharex=ax1, sharey=ax1)
    plt.title('(b) Dirty map', color='k', fontsize=17)
    plt.axis('off')
    print(f'lr range: [{np.min(datalr)}, {np.max(datalr)}]')
    lr = datalr[0, ..., 0]
    im2 = plt.imshow(lr, cmap=cmap, vmax=1, vmin=-1, 
               aspect='auto', extent=[0,1,0,1])
    plt.setp(ax2.spines.values(), color='k')
    plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, fraction=0.05)
    
    ax3 = plt.subplot(1,nsub,3, sharex=ax1, sharey=ax1)
    plt.title('(c) Reconstruction', c='C2', fontsize=17)
    sr = datasr[0, ...]
    print(f'sr range: [{np.min(sr)}, {np.max(sr)}]')
    im3 = plt.imshow(sr, cmap=cmap, vmax=1, vmin=-1, 
              aspect='auto', extent=[0,1,0,1])
    plt.axis('off')
    plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, fraction=0.05)


    ax4 = plt.subplot(1,nsub,4, sharex=ax1, sharey=ax1)
    plt.title('(d) Aleatoric Uncertainty', c='C2', fontsize=17)
    sr_au = datasr[1, ...]
    print(f'Aleatoric range: [{np.min(sr_au)}, {np.max(sr_au)}]')
    im4 = plt.imshow(sr_au, cmap='jet', 
              aspect='auto', extent=[0,1,0,1])
    plt.axis('off')
    plt.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.05, fraction=0.05)

    ax5 = plt.subplot(1,nsub,5, sharex=ax1, sharey=ax1)
    plt.title('(e) Epistemic Uncertainty', c='C2', fontsize=17)
    sr_eu = datasr[2, ...]
    print(f'Epistemic range: [{np.min(sr_eu)}, {np.max(sr_eu)}]')
    im5 = plt.imshow(sr_eu, cmap='jet', 
              aspect='auto', extent=[0,1,0,1])
    plt.axis('off')
    plt.colorbar(im5, ax=ax5, orientation='horizontal', pad=0.05, fraction=0.05)

    ax6 = plt.subplot(1,nsub,6, sharex=ax1, sharey=ax1)
    plt.title('(f) Absolute Error', c='C3', fontsize=17)
    ae = np.abs(datahr.squeeze() - sr)
    print(f'Absolute Error range: [{np.min(ae)}, {np.max(ae)}]')
    im6 = plt.imshow(ae, cmap='jet', 
              aspect='auto', extent=[0,1,0,1])
    plt.axis('off')
    plt.colorbar(im6, ax=ax6, orientation='horizontal', pad=0.05, fraction=0.05)

    plt.tight_layout()
    plt.show()
    plt.savefig('reconstruction.png')
    # plt.savefig('reconstruction.pdf')
    plt.close()

    precision_recall(datahr.squeeze(), sr, sr_au, sr_eu)
    calibration(datahr.squeeze(), sr, sr_au, sr_eu)


def reconstruct(model, fn_HR, T=50):
    # Get corresponding LR file
    fn_LR = fdirinLR + fn_HR.split('/')[-1][:-4]+'x%d.npy'%args.scale

    # Load files
    datahr = np.load(fn_HR)
    datalr = np.load(fn_LR)
    datalr = datalr[None,:,:]
    
    # Get mean and uncertainty predictions
    mu, au, eu = bnn_output(model, datalr, T=T)

    # Calculate PSNR and SSIM
    psnr = psnr_float(datahr, mu[..., None])
    ssim = ssim_float(tf.convert_to_tensor(datahr, dtype=tf.float32), tf.convert_to_tensor(mu[..., None]))

    # Calculate calibration
    _, freq_a, freq_e = calibration(datahr.squeeze(), mu, au, eu)
    
    return datahr, datalr, mu, au, eu, psnr, ssim, freq_a, freq_e


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog="analyze.py",
                                   description="Convert npy to fits mainly.")
    parser.add_argument('indir', type=str, help='In directory')
    parser.add_argument('fn_model', type=str, help='Model .h5 filename')
    parser.add_argument('-r', '--scale', dest='scale', type=int,
                      help="Upsample factor", default=4)
    parser.add_argument('-T', '--num_samples', type=int,
                      help="Number of Monte Carlo samples", default=20)
    parser.add_argument('-N', '--num_images', type=int,
                      help="Number of images to analyze", default=5)
    parser.add_argument('-o', '--out_dir', type=str, help='Directory to output to', default='')
    

    args = parser.parse_args()
    indir = args.indir
    fn_model = args.fn_model

    model = wdsr_bnn(scale=args.scale, num_res_blocks=32)
    model.load_weights(fn_model)

    if args.out_dir != '':
        fdiroutPLOT = args.out_dir
    else:
        fdiroutPLOT= indir+'/plot/'
    fdirinHR = indir+'/POLISH_valid_HR/'
    fdirinLR = indir+'/POLISH_valid_LR_bicubic/X%d/'%args.scale

    if not os.path.isdir(fdiroutPLOT):
        print("Making output plot directory")
        os.system('mkdir -p %s' % fdiroutPLOT)
    
    N = args.num_images

    # List of HR validation files (only use the first N)
    # flvalid = glob.glob(fdirinHR + '/*.npy')[-N:]
    flvalid = glob.glob(fdirinHR + '/*.npy')[:N]


    HR = np.zeros((N, 1800, 1800))
    LR = np.zeros((N, 600, 600))
    MU = np.zeros((N, 1800, 1800))
    AU = np.zeros((N, 1800, 1800))
    EU = np.zeros((N, 1800, 1800))
    PSNR = np.zeros(N)
    SSIM = np.zeros(N)
    FREQ_A = np.zeros((N, 50))
    FREQ_E = np.zeros((N, 50))
    
    CMAP ='afmhot'

    for i, fn_HR in enumerate(flvalid):
        hr, lr, mu, au, eu, psnr, ssim, freq_a, freq_e = reconstruct(model, fn_HR, T=args.num_samples)
        HR[i] = hr.squeeze()
        LR[i] = lr.squeeze()
        MU[i] = mu
        AU[i] = au
        EU[i] = eu
        PSNR[i] = psnr
        SSIM[i] = ssim
        FREQ_A[i] = freq_a
        FREQ_E[i] = freq_e
    
    fig, axs = plt.subplots(N, 6, figsize=(28, 4.25 * N))

    if N == 1:
        axs = axs[np.newaxis, ...]
    

    for i in range(N):
        axins = []

        # Plot ground truth, input, and predicted mean
        im0 = axs[i, 0].imshow(HR[i], cmap=CMAP, vmin=-1, vmax=1)
        fig.colorbar(im0, cax=make_axes_locatable(axs[i, 0]).append_axes('right', size='5%', pad=0.05))
        axins0 = zoomed_inset_axes(axs[i, 0], zoom=6, loc=4)
        axins0.imshow(HR[i], cmap=CMAP, vmin=-1, vmax=1)
        axins.append(axins0)

        im1 = axs[i, 1].imshow(LR[i], cmap=CMAP, vmin=-1, vmax=1)
        fig.colorbar(im1, cax=make_axes_locatable(axs[i, 1]).append_axes('right', size='5%', pad=0.05))
        axins1 = zoomed_inset_axes(axs[i, 1], zoom=6, loc=4)
        axins1.imshow(LR[i], cmap=CMAP, vmin=-1, vmax=1)
        axins.append(axins1)

        im2 = axs[i, 2].imshow(MU[i], cmap=CMAP, vmin=-1, vmax=1)
        fig.colorbar(im2, cax=make_axes_locatable(axs[i, 2]).append_axes('right', size='5%', pad=0.05))
        axins2 = zoomed_inset_axes(axs[i, 2], zoom=6, loc=4)
        axins2.imshow(MU[i], cmap=CMAP, vmin=-1, vmax=1)
        axins.append(axins2)

        # Plot uncertainties and absolute error
        im3 = axs[i, 3].imshow(AU[i], cmap='jet')
        fig.colorbar(im3, cax=make_axes_locatable(axs[i, 3]).append_axes('right', size='5%', pad=0.05))
        axins3 = zoomed_inset_axes(axs[i, 3], zoom=6, loc=4)
        axins3.imshow(AU[i], cmap='jet')
        axins.append(axins3)

        im4 = axs[i, 4].imshow(EU[i], cmap='jet')
        fig.colorbar(im4, cax=make_axes_locatable(axs[i, 4]).append_axes('right', size='5%', pad=0.05))
        axins4 = zoomed_inset_axes(axs[i, 4], zoom=6, loc=4)
        axins4.imshow(EU[i], cmap='jet')
        axins.append(axins4)

        im5 = axs[i, 5].imshow(np.abs(HR[i] - MU[i]), cmap='jet')
        fig.colorbar(im5, cax=make_axes_locatable(axs[i, 5]).append_axes('right', size='5%', pad=0.05))
        axins5 = zoomed_inset_axes(axs[i, 5], zoom=6, loc=4)
        axins5.imshow(np.abs(HR[i] - MU[i]), cmap='jet')
        axins.append(axins5)

        for j, axin in enumerate(axins):
            if j == 1:
                axin.set_xlim(875/3, 975/3)
                axin.set_ylim(400/3, 300/3)
            else:
                axin.set_xlim(875, 975)
                axin.set_ylim(400, 300)

            axin.yaxis.get_major_locator().set_params(nbins=7)
            axin.xaxis.get_major_locator().set_params(nbins=7)
            axin.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, left=False, right=False)
            for spine in axin.spines.values():
                spine.set_edgecolor('white')
            mark_inset(axs[i, j], axin, loc1=1, loc2=2, fc="none", ec="white")
    
    for ax in axs.ravel():
        ax.set_axis_off()

    for ax in axs.ravel():
        ax.set_axis_off()

    axs[0, 0].set_title('(a) Ground Truth', wrap=True)
    axs[0, 1].set_title('(b) Input', wrap=True)
    axs[0, 2].set_title('(c) Predicted', wrap=True)
    axs[0, 3].set_title('(d) Aleatoric Uncertainty', wrap=True)
    axs[0, 4].set_title('(e) Epistemic Uncertainty', wrap=True)
    axs[0, 5].set_title('(f) Absolute Error', wrap=True)

    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.1)
    # plt.tight_layout()

    plt.savefig(fdiroutPLOT + 'reconstruction.png')

    im = np.asarray(Image.open(fdiroutPLOT + 'reconstruction.png'))
    gamma = 1 / 2.2
    im = im ** gamma
    im = im / np.max(im) * 255
    im = Image.fromarray(np.uint8(im))
    im.save(fdiroutPLOT + 'reconstruction_gamma.png')


# if __name__=='__main__':
#     # Example usage:
#     # Generate images on training data:
#     # for im in ./images/PSF-nkern64-4x/train/X4/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done
#     # Generate images on validation data
#     # for im in ./images/PSF-nkern64-4x/valid/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done

#     parser = optparse.OptionParser(prog="hr2lr.py",
#                                    version="",
#                                    usage="%prog image weights.h5  [OPTIONS]",
#                                    description="Take high resolution images, deconvolve them, \
#                                    and save output.")

#     parser.add_option('-f', dest='fnhr', 
#                       help="high-res file name", default=None)
#     parser.add_option('-x', dest='scale', type=int,
#                       help="spatial rebin factor", default=4)
#     parser.add_option('-b', '--nbit', dest='nbit', type=int,
#                       help="number of bits in image", default=16)
#     parser.add_option('-p', '--plotit', dest='plotit', action="store_true",
#                       help="plot")

#     options, args = parser.parse_args()
#     fn_img, fn_model = args

#     datalr, datasr, datahr = reconstruct(fn_img, fn_model, options.scale, 
#                                  fnhr=options.fnhr, nbit=options.nbit)

#     if datahr is not None:
#         nsub = 6 
#     else:
#         nsub = 5

#     if options.plotit:
#         plot_reconstruction(datalr, datasr, datahr=datahr, vm=1, nsub=nsub)







