import argparse
import pickle

import seaborn as sns 
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def match(gt, pred, thresh=10):
    print('\nCross validating source catalogs...')
    unmatched_gt = np.ones(gt.shape[0], dtype=bool)
    unmatched_pred = np.ones(pred.shape[0], dtype=bool)
    matches = np.ones(gt.shape[0]) * -1

    dist = cdist(gt, pred)  # The distance from each gt to each predicted source

    for _ in range(min(gt.shape[0], pred.shape[0])):
        # mask out previously matched sources
        masked = np.where(unmatched_gt[:, np.newaxis] & unmatched_pred[np.newaxis, :], dist, np.inf)

        # Find the closest source for each in the gt
        closest = masked.argmin(axis=1)
        closest_dist = masked.min(axis=1)

        # If none of the sources are close we're done
        if closest_dist.min() > thresh:
            break

        # Else, the next one to match is the gt with the closest source
        i = np.argmin(closest_dist)
        matches[i] = closest[i]

        # Update mask
        unmatched_gt[i] = 0
        unmatched_pred[closest[i]] = 0
    
    return matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fnsr', type=str, help='Super resolution fits file name')
    parser.add_argument('fnsx', type=str, help='SR SExtractor file name')
    parser.add_argument('fntr', type=str, help='True SExtractor file')
    parser.add_argument('--outname', '-o', type=str, help='Output file name',
                        default='temp.png')

    # gt = np.array([[0, 0], [2, 0], [3, 0]])
    # gt = np.array([[0, 0], [2, 0], [3, 0], [4, 0]])
    # pred = np.array([[0, 1], [2, 1], [3, 1]])
    # pred = np.array([[3, 1], [2, 1], [0, 1]])
    # pred = np.array([[3, 1], [2, 1], [0, 1]])
    # print(match(gt, pred))

    args = parser.parse_args()

    datasr = fits.getdata(args.fnsr)
    sx = np.genfromtxt(args.fnsx)

    fig = plt.figure(figsize=(5, 5))

    plt.imshow(datasr, vmax=22500/20)

    t = np.genfromtxt(args.fntr)
    print(f'Number of original sources: {t.shape[0]}')
    plt.scatter(t[..., 3], t[..., 4],  c='blue', label='True sources')

    print(f'Number of SExtracted sources: {sx.shape[0]}')
    plt.scatter(sx[..., 3], sx[..., 4],  c='red', label='Detected sources')

    matches = match(t[..., 3:5], sx[..., 3:5])

    matched_t = t[np.where(matches >= 0)]
    matched_sx = sx[matches[matches >= 0].astype(int)]
    for i in range(matched_t.shape[0]):
        plt.plot([matched_t[i, 3], matched_sx[i, 3]], [matched_t[i, 4], matched_sx[i, 4]], 'k-')

    print(f'\nNumber of matched sources: {matched_t.shape[0]}')
    missed_sources = t.shape[0] - len(matched_t)
    extra_sources = sx.shape[0] - len(matched_t)
    if missed_sources >= 0:
        print(f'Number of missed sources: {missed_sources}')
    if extra_sources > 0:
        print(f'Number of hallucinated sources: {extra_sources}')

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(args.outname)
    pickle.dump(fig, open(args.outname.split('.')[-2]+'.pickle', 'wb'))
    plt.close()

    # cm = np.array([[matched_t.shape[0], missed_sources], [extra_sources, 0]])
    cm = np.array([[185, 31], [17, 0]])
    xcategories = ['Detected', 'Missed']
    fig = plt.figure(figsize=(3, 3))
    ycategories = ['Positive', 'Negative']
    sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues', cbar=False,
            xticklabels=xcategories, yticklabels=ycategories)
    plt.ylabel('True Label')
    plt.xlabel('SExtractor Status')
    plt.tight_layout()
    plt.show()
    plt.savefig('confusion.png')

