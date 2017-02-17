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
np.random.shuffle(input_images)

print 'Testing - shot noise limited?'
npts = 200
# len_t = len(input_images)
# len_x = input_images[0].shape[0]
# len_y = input_images[0].shape[1]
# print  len_t, len_x,  len_y

# How does frame intensity vary with t?
noise_intensity = bn.nanmean(bn.nanmean(input_images, axis=1), axis=1)
#plt.subplot(1, 2, 1)
#plt.plot(noise_intensity)
#plt.ylabel('noise_intensity')

# How does frame standard devaiation vary with t?
# frame_stdev = bn.nanstd(bn.nanstd(input_images, axis=1), axis=1)
# noise_std = np.empty(len(input_images))
# noise_std[0:len(input_images)] =0
# for n in tqdm(range(1,50)):
#    noise_std[n-1] =  bn.nanstd(input_images[0:n+1])

# How does standard deviation of a frame vary with number of frames group averaged together?
# And how does the difference in pixel intensity between frames vary with number of frames group averaged together?
# curtailed length of output array by a factor of 4 to speed this up.
noise_std_groupmean = np.empty(npts)
noise_dif_groupmean = np.empty(npts)
# print bn.nanmean(input_images[0:1], axis=0)
# print bn.nanmean(input_images[2:3], axis=0)

# print bn.nanmean(np.subtract(bn.nanmean(input_images[0:1], axis=0),bn.nanmean(input_images[2:3], axis=0)))
# for n in tqdm(range(1,len(input_images)/4)) :
for n in tqdm(range(0, npts)):
    frame = nFrames / npts * (n+1) - 1
    #print frame
    mean = bn.nanstd(bn.nanmean(input_images[0:frame], axis=0))
    noise_std_groupmean[n] = mean
    noise_dif_groupmean[n] = bn.nanstd(np.subtract(bn.nanmean(input_images[0:(frame+1)/2-1], axis=0),
                                                                bn.nanmean(input_images[(frame+1)/2:frame],
                                                                        axis=0)))
    #noise_dif_groupmean[n]



#Plotting
ypts = nFrames/ npts * (np.arange(npts) +1)

with open(filename +'_shuffled_out.csv', 'wb') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerows([ypts, noise_std_groupmean, ypts/2, noise_dif_groupmean])

plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
#plt.plot(ypts, noise_std_groupmean, 'ro', label='noise_std_groupmean')
#plt.plot(ypts/2, noise_dif_groupmean, 'ko', label='noise_dif_groupmean')

#plt.ylim([np.min(noise_std_groupmean), np.max(noise_std_groupmean)])
#plt.ylim([nFrames / npts * (1) - 1, nFrames / npts * (npts+1) - 1])

plotmin = np.min(noise_dif_groupmean)
plotmax = np.max(noise_dif_groupmean)
#plt.xscale('log')
plt.yscale('log')
plt.ylabel('Log standard deviation, difference method')
plt.plot(ypts/2,noise_dif_groupmean, 'ro', label='noise_dif_groupmean')
plt.ylim([plotmin, plotmax])

#plt.plot(ypts, noise_std_groupmean, 'ro', label='noise_std_groupmean')
#plt.plot(ypts/2, noise_dif_groupmean, 'ko', label='noise_dif_groupmean')
plt.ylim([plotmin, plotmax])
plt.xlim([ypts[0]/2,ypts[npts-1]/2])
#plt.ylim([np.min(noise_std_groupmean), np.max(noise_std_groupmean)])
#plt.ylim([nFrames / npts * (1) - 1, nFrames / npts * (npts+1) - 1])


plt.subplot(1,2,2)

plt.xscale('log')
plt.yscale('log')
plt.ylabel('Log standard deviation, difference method')
plt.plot(ypts/2,noise_dif_groupmean, 'ko', label='noise_dif_groupmean')
plt.ylim([plotmin, plotmax])
plt.xlim([ypts[0]/2,ypts[npts-1]/2])
# plt.legend(loc='upper center')
# Finish up


plt.suptitle(filename+' shuffled')
plt.show()


print "Finished Testing!"
quit()