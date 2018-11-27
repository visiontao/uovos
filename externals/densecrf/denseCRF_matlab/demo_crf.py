import pydensecrf.densecrf as dcrf
import numpy as np
import sys

from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

from os import listdir, makedirs
from os.path import isfile, join


img = imread('00006.jpg') 
anno_rgb = imread('00006_anno.jpg').astype(np.uint32)

min_val = np.min(anno_rgb.ravel())
max_val = np.max(anno_rgb.ravel())
out = (anno_rgb.astype('float') - min_val) / (max_val - min_val)
labels = np.zeros((2, img.shape[0], img.shape[1]))
labels[1, :, :] = out
labels[0, :, :] = 1 - out

colors = [0, 255]
colorize = np.empty((len(colors), 1), np.uint8)
colorize[:,0] = colors

n_labels = 2

crf = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

U = unary_from_softmax(labels)
crf.setUnaryEnergy(U)

feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
crf.addPairwiseEnergy(feats, compat=3,
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC)

# This creates the color-dependent features and then add them to the CRF
feats = create_pairwise_bilateral(sdims=(50, 50), schan=(10, 10, 10),
                              img=img, chdim=2)
crf.addPairwiseEnergy(feats, compat=5,
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC)

Q = crf.inference(5)

MAP = np.argmax(Q, axis=0)
MAP = colorize[MAP]

imsave('seg.jpg', MAP.reshape(anno_rgb.shape))
