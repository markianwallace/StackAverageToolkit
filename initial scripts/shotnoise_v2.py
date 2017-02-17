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
parser.add_argument('--norm', action='store_true', help='norm each frame')
args = parser.parse_args()

filename = args.filename

input_images = pims.open(args.filename)
input_images = np.squeeze(input_images)
nFrames = len(input_images)

#Crop of bottom row of pixels for analysis, which on the photon focus contain metadata
input_images[:,-1,:] = np.nan



print 'Testing - shot noise limited?'
npts = 30
# len_t = len(input_images)
# len_x = input_images[0].shape[0]
# len_y = input_images[0].shape[1]
# print  len_t, len_x,  len_y

# How does frame intensity vary with t?
noise_intensity = bn.nanmean(bn.nanmean(input_images, axis=1), axis=1)
#plt.subplot(1, 2, 1)
#plt.plot(noise_intensity)
#plt.ylabel('noise_intensity')

if args.norm:
    print 'norming'
    for n in tqdm(np.arange(len(input_images))):
        input_images[n] = input_images[n]/np.nansum(input_images[n])


# How does standard deviation of a frame vary with number of frames group averaged together?
# And how does the difference in pixel intensity between frames vary with number of frames group averaged together?
# curtailed length of output array by a factor of 4 to speed this up.
noise_std_groupmean = np.empty(npts)
noise_dif_groupmean = np.empty(npts)
mean_intensity = np.empty(npts)
# print bn.nanmean(input_images[0:1], axis=0)
# print bn.nanmean(input_images[2:3], axis=0)

# print bn.nanmean(np.subtract(bn.nanmean(input_images[0:1], axis=0),bn.nanmean(input_images[2:3], axis=0)))
# for n in tqdm(range(1,len(input_images)/4)) :
for n in tqdm(range(0, npts)):
    frame = nFrames / npts * (n+1) - 1
    #print frame
    #mean = bn.nanstd(bn.nanmean(input_images[0:frame], axis=0))
    mean_intensity[n] = bn.nanmean(input_images[frame])
    noise_std_groupmean[n] = bn.nanstd(bn.nanmean(input_images[0:frame], axis=0))
    noise_dif_groupmean[n] = bn.nanstd(np.subtract(bn.nanmean(input_images[0:(frame+1)/2-1], axis=0),
                                                                bn.nanmean(input_images[(frame+1)/2:frame],
                                                                        axis=0)))
    #noise_dif_groupmean[n]
skippedFrames = 0

for n in tqdm(np.arange(len(input_images)-1)):
    #print n
    #print np.nansum(input_images[n] - input_images[n+1])
    #print input_images[n + 1]
    if np.nansum(input_images[n] - input_images[n+1]) == 0:

        skippedFrames += 1
print 'finished calculations'


#Plotting
ypts = nFrames/ npts * (np.arange(npts) +1)

plt.figure(figsize = (15, 5))

plt.subplot(1, 3, 1)
plt.ylabel('Mean Px Intensity')
plt.xlabel('Frame')
plt.plot(ypts/2,mean_intensity, 'ro', label='noise_dif_groupmean')


plt.subplot(1, 3, 2)
#plt.plot(ypts, noise_std_groupmean, 'ro', label='noise_std_groupmean')
#plt.plot(ypts/2, noise_dif_groupmean, 'ko', label='noise_dif_groupmean')

#plt.ylim([np.min(noise_std_groupmean), np.max(noise_std_groupmean)])
#plt.ylim([nFrames / npts * (1) - 1, nFrames / npts * (npts+1) - 1])

plotmin = np.min(noise_dif_groupmean)
plotmax = np.max(noise_dif_groupmean)
plt.xscale('log')
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


plt.subplot(1,3,3)

plt.xscale('log')
plt.yscale('log')
plt.ylabel('Log standard deviation, difference method')
plt.plot(ypts/2,noise_dif_groupmean, 'ko', label='noise_dif_groupmean')
plt.ylim([plotmin, plotmax])
plt.xlim([ypts[0]/2,ypts[npts-1]/2])
# plt.legend(loc='upper center')
# Finish up


with open(filename +"_"+str(skippedFrames)+'skipped_out.csv', 'wb') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerows([ypts, noise_std_groupmean, ypts/2, noise_dif_groupmean])
plt.suptitle(filename + ' ' + str(skippedFrames)+' skipped frames')
plt.show()

print "skipped frames:"
print skippedFrames
print "Finished Testing!"
quit()