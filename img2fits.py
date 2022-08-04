import os 

import argparse
import numpy as np
import glob 
from astropy.io import fits
import matplotlib.pylab as plt

from utils import load_image, plot_sample
from model.wdsr import wdsr_b
from model.common import resolve_single, tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

fn_fits_example = './data-examples/fits/random-60x15s-with-noise.fits'

def img2fits(fn_img, fn_fits_example, fn_fitsout):
    if type(fn_img)==str:
        data = load_image(fn_img)
    elif type(fn_img)==np.ndarray:
        data = fn_img
    else:
        print("Expected image or np.array input")
        print(fn_img)
        exit()

    f = fits.open(fn_fits_example)
    header_example = f[0].header
    hdu = fits.PrimaryHDU(data=data, header=header_example)
    hdu.writeto(fn_fitsout)

if __name__=='__main__':
    # Example usage:
    # Generate images on training data:
    # for im in ./images/PSF-nkern64-4x/train/X4/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done
    # Generate images on validation data
    # for im in ./images/PSF-nkern64-4x/valid/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done

    parser = argparse.ArgumentParser(prog="img2fits.py",
                                   description="Convert png to fits mainly.")
    parser.add_argument('indir', type=str, help='In directory')
    parser.add_argument('fn_model', type=str, help='Model .h5 filename')
    parser.add_argument('-r', '--scale', dest='scale', type=int,
                      help="Upsample factor", default=4)

    args = parser.parse_args()
    indir = args.indir
    fn_model = args.fn_model

    model = wdsr_b(scale=args.scale, num_res_blocks=32)
    model.load_weights(fn_model)

    # fdiroutTRAIN = indir+'/train/'
    fdiroutVALID = indir+'/valid/'
    fdiroutFITS = indir+'/fits/'

    if not os.path.isdir(fdiroutFITS):
        print("Making output fits directory")
        os.system('mkdir -p %s' % fdiroutFITS)
    
    # fltrain = glob.glob(fdiroutTRAIN+'/*png')
    flvalid = glob.glob(fdiroutVALID+'/*png')

    # fltrain = glob.glob(indir+'/POLISH_train_HR/*.png') + glob.glob(indir+'/POLISH_train_HR/x%d/*.png'%args.scale)
    flvalid = glob.glob(indir+'/POLISH_valid*/*.png') + glob.glob(indir+'/POLISH_valid*/X%d/*.png'%args.scale)
    print(flvalid)
    
    for fn in flvalid:
        if 'x%d'%args.scale in fn:
            fn_fitsout_SR = fdiroutFITS + fn.split('/')[-1].strip('x%d.png'%args.scale)+'SR.fits'
            if os.path.exists(fn_fitsout_SR):
                continue
            datalr = load_image(fn)
            datalr = datalr[:,:,None]
            datasr = resolve_single(model, datalr)
            datasr = datasr.numpy()[..., 0]
            img2fits(datasr, fn_fits_example, fn_fitsout_SR)
            continue

        fn_fitsout_HR = fdiroutFITS + fn.split('/')[-1].strip('.png')+'.fits'
        # fn_fitsout_HRnoise = fdiroutFITS + fn.split('/')[-1].strip('.png')+'noise.fits'        
        # fn_fitsout_LR = fdiroutFITS + fn.split('/')[-1].strip('.png')+'x%d.fits'%args.scale
        if os.path.exists(fn_fitsout_HR):
            continue
        # if os.path.exists(fn_fitsout_LR):
        #     continue
        data = load_image(fn)
        img2fits(data, fn_fits_example, fn_fitsout_HR)
        # print("Wrote to fits:\n%s"%fn_fitsout_HR)
        # img2fits(dataLR, fn_fits_example, fn_fitsout_LR)
        # img2fits(datahr_noise, fn_fits_example, fn_fitsout_HRnoise)    