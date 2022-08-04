import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
from astropy.io import fits

from reconstruct import reconstruct
# import hr2lr


cmaps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
if 'afmhot_10us' in cmaps:
    cmap_global = 'afmhot_10us'
else:
    print("Could not load afmhot_10us, using afmhot")
    cmap_global = 'afmhot'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def plot(model_name, lr, sr, hr):
    import matplotlib.patches as patches

    print(lr.shape)
    print(sr.shape)
    print(hr.shape)

    # lr = np.load('./plots/ska-fun-mid-dirty-625.npy')
    # sr = np.load('./plots/ska-fun-mid-SR-1875.npy')
    # hr = np.load('./plots/ska-fun-mid-true-1875.npy')
    # cr = np.load('./plots/ska-fun-mid-clean-1875.npy')
    # D = cr

    et = [0,1,1,0]


    # hr = hr2lr.normalize_data(hr)
    # sr = hr2lr.normalize_data(sr)
    # D = hr2lr.normalize_data(D)
    # DD = D - np.median(D)
    # DD[DD<0] = 0

    nbit = 16
    # psnr_clean = tf.image.psnr(DD[None, ...,None].astype(np.uint16),
    #                            hr[None, ..., None].astype(np.uint16),
    #                            max_val=2**(nbit)-1)
    psnr_polish = tf.image.psnr(tf.convert_to_tensor(sr[None, ...].astype(np.uint16)),
                               hr[None, ..., None].astype(np.uint16),
                               max_val=2**(nbit)-1)

    # ssim_clean = tf.image.ssim(DD[None, ...,None].astype(np.uint16),
    #                           hr[None, ..., None].astype(np.uint16),
    #                            max_val=2**(nbit)-1)
    ssim_polish = tf.image.ssim(tf.convert_to_tensor(sr[None, ...].astype(np.uint16)),
                               tf.convert_to_tensor(hr[None, ..., None].astype(np.uint16)),
                               max_val=2**(nbit)-1)

    # clean_box = "PSNR = %0.1f\nSSIM = %0.3f" % (psnr_clean, ssim_clean)
    polish_box = "PSNR = %0.1f\nSSIM = %0.3f" % (psnr_polish, ssim_polish)

    plt.figure(figsize=(20,12))

    gs = gridspec.GridSpec(2, 8)

    lr = lr - lr.min()
    lr = lr / lr.max()

    sr = sr - sr.min()
    sr = sr / sr.max()

    hr = hr - hr.min()
    hr = hr / hr.max()

    # D = D - D.min()
    # D = D / D.max()

    a,b,c,d = 330, 820, 430, 930
    ax1 = plt.subplot(gs[0, 0:2])
    vmx = hr[910:1050, 810:974].max()
    vmn = hr[910:1050, 810:974].min()

    ax1.imshow(lr**0.85, cmap=cmap_global, vmax=vmx, vmin=vmn, )
    # Create a Rectangle patch
    rect1 = patches.Rectangle((810//3, 910//3), 164//3, 140//3, linewidth=3, edgecolor='C1', facecolor='none')
    rect2 = patches.Rectangle((a//3, b//3), 100//3, 100//3, linewidth=3, edgecolor='C1', facecolor='none')
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    plt.axis('off')
    plt.title('Dirty image', c='C1', fontsize=31)


    ax2 = plt.subplot(gs[0, 4:6])
    ax2.imshow(sr**0.85, cmap=cmap_global, vmax=vmx, vmin=vmn,  )
    # Create a Rectangle patch
    rect1 = patches.Rectangle((810, 910), 164, 140, linewidth=3, edgecolor='C2', facecolor='none')
    rect2 = patches.Rectangle((a, b), 100, 100, linewidth=3, edgecolor='C2', facecolor='none')
    ax2.add_patch(rect1)
    ax2.add_patch(rect2)
    plt.title('POLISH reconstruction', c='C2', fontsize=31)

    props = dict(facecolor='k', alpha=0., edgecolor='k')
#    plt.text(0.35*len(hr), 0.18*len(hr), polish_box, color='white', fontsize=28, 
#          fontweight='bold', bbox=props)

    ax3 = plt.subplot(gs[0, 6:8])
    immy = ax3.imshow(hr**0.85, vmax=vmx, vmin=vmn, cmap=cmap_global)
    rect1 = patches.Rectangle((810, 910), 164, 140, linewidth=3, edgecolor='grey', facecolor='none')
    rect2 = patches.Rectangle((a, b), 100, 100, linewidth=3, edgecolor='grey', facecolor='none')
    ax3.add_patch(rect1)
    ax3.add_patch(rect2)
    #axis('off')
    plt.title('True sky', c='k', fontsize=31)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(immy,cax=cax)

    ax4 = plt.subplot(gs[0, 2:4])
    # ax4.imshow(D**0.85, vmax=vmx, vmin=0.05, cmap=cmap_global)
    # Create a Rectangle patch
    rect1 = patches.Rectangle((810, 910), 164, 140, linewidth=3, edgecolor='C0', facecolor='none')
    rect2 = patches.Rectangle((a, b), 100, 100, linewidth=3, edgecolor='C0', facecolor='none')
    ax4.add_patch(rect1)
    ax4.add_patch(rect2)
    plt.axis('off')
    plt.title('CLEAN reconstruction', c='C0', fontsize=31)

#        plt.text(xlim2+0.015, ylim2+0.005, psnr_2, color='C3', fontsize=9, fontweight='bold')
#        props = dict(boxstyle='', facecolor='white', alpha=0.25, edgecolor='white')

   # plt.text(0.35*len(hr), 0.18*len(hr), clean_box, color='white', fontsize=28, 
   #        fontweight='bold', bbox=props)

    lr_ = lr[b//3:d//3, a//3:c//3]
    sr_ = sr[b:d, a:c]
    hr_ = hr[b:d, a:c]
    # D_ = D[b:d, a:c]

    lr_ = lr_ - lr_.min()
    lr_ = lr_ / lr_.max()

    sr_ = sr_ - sr_.min()
    sr_ = sr_ / sr_.max()

    hr_ = hr_ - hr_.min()
    hr_ = hr_ / hr_.max()

    # D_ = D_ - D_.min()
    # D_ = D_ / D_.max()

    ax5 = plt.subplot(gs[1, 0])
    ax5.imshow(lr_**0.85, cmap=cmap_global,extent=et, vmax=0.5, vmin=0, )
    plt.xticks([])
    plt.yticks([])

    ax6 = plt.subplot(gs[1, 4])
    ax6.imshow(sr_**0.85, cmap=cmap_global, vmax=0.5, vmin=0,  extent=et)
    plt.xticks([])
    plt.yticks([])
    
    ax7 = plt.subplot(gs[1, 6])
    ax7.imshow(hr_**0.85, vmax=0.5, vmin=0,  extent=et, cmap=cmap_global)
    plt.xticks([])
    plt.yticks([])
    
    ax8 = plt.subplot(gs[1, 2])
    # ax8.imshow(D_**0.85, vmax=0.5, vmin=0, extent=et, cmap=cmap_global)
    plt.xticks([])
    plt.yticks([])

    lr_ = lr[910//3:1050//3, 810//3:974//3]
    sr_ = sr[910:1050, 810:974]
    hr_ = hr[910:1050, 810:974]
    # D_ = D[910:1050, 810:974]

    lr_ = lr_ - lr_.min()
    lr_ = lr_ / lr_.max()

    sr_ = sr_ - sr_.min()
    sr_ = sr_ / sr_.max()

    hr_ = hr_ - hr_.min()
    hr_ = hr_ / hr_.max()

    # D_ = D_ - D_.min()
    # D_ = D_ / D_.max()

    ax9 = plt.subplot(gs[1, 1])
    ax9.imshow(lr_**0.85, cmap=cmap_global,extent=et, vmax=0.5, vmin=0, )
    plt.xticks([])
    plt.yticks([])

    ax10 = plt.subplot(gs[1, 5])
    ax10.imshow(sr_**0.85, cmap=cmap_global, vmax=0.5, vmin=0,  extent=et)
    plt.xticks([])
    plt.yticks([])

    ax11 = plt.subplot(gs[1, 7])
    ax11.imshow(hr_**0.85, vmax=0.5, vmin=0,  extent=et, cmap=cmap_global)
    plt.xticks([])
    plt.yticks([])
    
    ax12 = plt.subplot(gs[1, 3])
    # ax12.imshow(D_**0.85, vmax=0.5, vmin=0, extent=et, cmap=cmap_global)
    plt.xticks([])
    plt.yticks([])

    plt.setp(ax5.spines.values(), color='C1', lw=5)
    plt.setp(ax9.spines.values(), color='C1', lw=5)
    plt.setp(ax7.spines.values(), color='grey', alpha=1, lw=5)
    plt.setp(ax11.spines.values(), color='grey', alpha=1, lw=5)
    plt.setp(ax6.spines.values(), color='C2', lw=5)
    plt.setp(ax10.spines.values(), color='C2', lw=5)
    plt.setp(ax8.spines.values(), color='C0', lw=5)
    plt.setp(ax12.spines.values(), color='C0', lw=5)

    plt.savefig(f'plot/{model_name}-plot.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create Visualizations for POLISH Output')
    parser.add_argument('image', type=str, help='Image to reconstruct and plot')
    parser.add_argument('fnhr', type=str, help="high-res file name", default=None)
    parser.add_argument('model', type=str, help='Model .h5 file')
    parser.add_argument('-x', dest='scale', 
                      help="spatial rebin factor", default=4)
    parser.add_argument('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits in image", default=16)
    parser.add_argument('-p', '--plotit', dest='plotit', action="store_true",
                      help="plot")
    # parser.add_argument(
    #     '--data_directory', '-d', type=str,
    #     help='Directory for data set. Defaults to one matching model name'
    # )

    args = parser.parse_args()


    datalr, datasr, datahr = reconstruct(args.image, args.model, args.scale, 
                                 fnhr=args.fnhr, nbit=args.nbit)
    
    plot(args.model[:-3], datalr, datasr, datahr)

