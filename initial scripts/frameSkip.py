import pims
import argparse
import bottleneck as bn
from skimage import io
import numpy as np
from tqdm import tqdm # adds a taskbar to any loop, e.g. for i in tqdm(range(10000)):
from matplotlib import pyplot as plt # for plotting
import os #for making directories
import csv

parser = argparse.ArgumentParser()

parser.add_argument('filename', help='Name of input file, or the directory containing images')
args = parser.parse_args()

filename = args.filename

input_images = pims.open(args.filename)
input_images = np.squeeze(input_images)
nFrames = len(input_images)

#Crop of bottom row of pixels for analysis, which on the photon focus contain metadata
input_images[:,-1,:] = np.nan


skippedFrames = 0

for n in tqdm(np.arange(len(input_images)-1)):
    #print n
    #print np.nansum(input_images[n] - input_images[n+1])
    #print input_images[n + 1]
    if np.nansum(input_images[n] - input_images[n+1]) == 0:

        skippedFrames += 1


print skippedFrames


quit()