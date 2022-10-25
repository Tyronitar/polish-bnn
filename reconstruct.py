import os
import sys

import optparse
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
from PIL import Image

from model import resolve_single, normalize_bnn, resolve_bnn, resolve_float
from utils import load_image, plot_sample
from model.wdsr import wdsr_b
from model.bnn import wdsr_bnn
import tensorflow as tf

plt.rcParams.update({
                    'font.size': 12,
                    'font.family': 'serif',
                    'axes.labelsize': 14,
                    'axes.titlesize': 15,
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

def reconstruct(fn_img, fn_model, scale, fnhr=None,
                nbit=16):
    if fn_img.endswith('npy'):
        datalr = np.load(fn_img)[:, :]
    elif fn_img.endswith('png'):
      try:
          datalr = load_image(fn_img)
      except:
          return 

    # datalr = tf.squeeze(normalize_bnn(datalr[None, ...]))

    print(f'\n\n{fnhr}\n\n')
    datahr = None
    if fnhr is not None:
        if fnhr.endswith('npy'):
            datahr = np.load(fnhr)[:, :]
        elif fnhr.endswith('png'):
          try:
              datahr = load_image(fnhr)
          except:
              return 
        # datahr = tf.squeeze(normalize_bnn(datahr[None, ...]))


    model = wdsr_bnn(scale=scale, num_res_blocks=32)
    model.load_weights(fn_model)
    datalr = datalr[None,:,:]




    # datasr = resolve_bnn(model, datalr)
    T = 15
    srs = np.stack([resolve_float(model, datalr).numpy().squeeze() for _ in range(T)])
    print(srs.shape)
    datasr = np.stack([np.mean(srs[..., 0], axis=0), np.mean(2 * np.exp(srs[..., 1]), axis=0), np.var(srs[..., 0], axis=0)])
    print(datasr.shape)
    # datasr = datasr.numpy()
    return datalr, datasr, datahr

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
    
    plt.figure(figsize=(8,5))
    plt.plot(ps / 100., precision_a, 'r', label='Aleatoric')
    plt.plot(ps / 100., precision_e, 'b--', label='Epistemic')
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision (RMSE)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('precision_recall.png')
    plt.close()


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
        
    plt.figure(figsize=(8,5))
    plt.plot(ps, freq_a, 'r', label='Aleatoric')
    plt.plot(ps, freq_e, 'b--', label='Epistemic')
    plt.plot([0,1],[0,1], 'k')
    plt.margins(x=0, y=0)
    plt.legend()
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('calibration.png')
    plt.close()


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
    axins = zoomed_inset_axes(ax1, 6, loc=1)
    axins.imshow(hr[50:100, 50:100], cmap=cmap, vmax=1, vmin=-1, 
            aspect='auto', extent=[0,1,0,1])
    axins.set_xlim(50, 100)
    axins.set_ylim(50, 100)
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

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

    im = np.asarray(Image.open('reconstruction.png'))
    gamma = 1 / 2.2
    im = im ** gamma
    im = im / np.max(im) * 255
    im = Image.fromarray(np.uint8(im))
    im.save('reconstruction_gamma.png')
    

    precision_recall(datahr.squeeze(), sr, sr_au, sr_eu)
    calibration(datahr.squeeze(), sr, sr_au, sr_eu)


if __name__=='__main__':
    # Example usage:
    # Generate images on training data:
    # for im in ./images/PSF-nkern64-4x/train/X4/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done
    # Generate images on validation data
    # for im in ./images/PSF-nkern64-4x/valid/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done

    parser = optparse.OptionParser(prog="hr2lr.py",
                                   version="",
                                   usage="%prog image weights.h5  [OPTIONS]",
                                   description="Take high resolution images, deconvolve them, \
                                   and save output.")

    parser.add_option('-f', dest='fnhr', 
                      help="high-res file name", default=None)
    parser.add_option('-x', dest='scale', type=int,
                      help="spatial rebin factor", default=4)
    parser.add_option('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits in image", default=16)
    parser.add_option('-p', '--plotit', dest='plotit', action="store_true",
                      help="plot")

    options, args = parser.parse_args()
    fn_img, fn_model = args

    datalr, datasr, datahr = reconstruct(fn_img, fn_model, options.scale, 
                                 fnhr=options.fnhr, nbit=options.nbit)

    if datahr is not None:
        nsub = 6 
    else:
        nsub = 5

    if options.plotit:
        plot_reconstruction(datalr, datasr, datahr=datahr, vm=1, nsub=nsub)







