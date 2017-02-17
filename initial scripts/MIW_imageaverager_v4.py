# Setup Instructions
# -----
# pip install bottleneck
# pip install scikit-image
# pip install pims
# pip install jpype1
# pip install tifffile
# pip install tqdm

# Example Useage
# -----
#python MIW_imageaverager.py --median --running --filterwindow 50 2PEG5us10short.cine
#python MIW_imageaverager.py --median --grouped --filterwindow 50 'test/*.tif'
#python MIW_imageaverager.py --median --grouped --filterwindow 50 percsim_f_0.1.tif

import pims
import argparse
import bottleneck as bn
from skimage import io
import numpy as np
from tqdm import tqdm # adds a taskbar to any loop, e.g. for i in tqdm(range(10000)):
from matplotlib import pyplot as plt # for plotting
import os #for making directories

# Get some variables from command line (nb switch raw_input for input in python v3 here it's v2.7)
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='mean flag')
parser.add_argument('--quicksave', action='store_true', help='quicksave flag')
parser.add_argument('--mean', action='store_true', help='mean flag')
parser.add_argument('--median', action='store_true', help='median flag' )
parser.add_argument('--running', action='store_true', help='running flag' )
parser.add_argument('--grouped', action='store_true', help='grouped flag' )
parser.add_argument('--filterwindow', type=int, help='The size of the filter window')
parser.add_argument('filename', help='Name of input file, or the directory containing images')
parser.add_argument('--normalise', action='store_true', help='normalise the data?' )
parser.add_argument('--manyfiles', type=int, help='save output as this many TIF stacks' )
args = parser.parse_args()
filterwindow = args.filterwindow
filename = args.filename


#Definition for pretty outputs
def prettyplot(x, y, title, xlabel, ylabel):
    # These imports might not be needed if done in rest of code. We should really import the modules at the top as usual, and then inside the function assign the module to these local variables.
    from matplotlib import pyplot as plt
    import seaborn as sns
    
    # General Settings
    sns.set(font_scale=1.5)
    sns.set_style('white')
    sns.set_style('ticks', {"xtick.major.size": 8, "ytick.major.size": 8})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.markeredgecolor = 'black'
    plt.tight_layout() # Fix up margins on the output.
    sns.despine() # Remove the border and the ticks on the top and the right hand side of the plot

    # Autoscale x axis to include a bit more space
    x_axisrange = abs(x.max()-y.min())
    x_minoffset = x.min()- 0.1*x_axisrange
    x_maxoffset = x.max()+ 0.1 * x_axisrange
    plt.xlim([x_minoffset,x_maxoffset])

    # Autoscale y axis to include a bit more space
    y_axisrange = abs(y.max()-y.min())
    y_minoffset = y.min()- 0.1*y_axisrange
    y_maxoffset = y.max()+ 0.1 * y_axisrange
    plt.ylim([y_minoffset,y_maxoffset])

    # Generate a plot to return
    plot_output = plt.plot(x,y, alpha=0.5, marker='o', markersize=8, ls='' , mew=1)
    
    #Optionally display an output
    plt.show()
    
    return plot_output




# open the file using the magic of PIMS.
input_images = pims.open(args.filename)

if args.quicksave :
    im_out = np.uint16(input_images)
    #io.imsave(filename + '.output.tif',im_out)

elif args.test:
    print 'Testing - shot noise limited?'
    #len_t = len(input_images)
    #len_x = input_images[0].shape[0]
    #len_y = input_images[0].shape[1]
    #print  len_t, len_x,  len_y
    
    # How does frame intensity vary with t?
    noise_intensity = bn.nanmean(bn.nanmean(input_images, axis=1), axis=1)
    t=range(0,len(input_images))
    plt.subplot(1,2,1)
    print t, noise_intensity
    prettyplot(t,noise_intensity, 'title', 'xlabel', 'ylabel')
    
    #plt.plot()
    plt.ylabel('noise_intensity')
    
    #How does frame standard devaiation vary with t?
    #frame_stdev = bn.nanstd(bn.nanstd(input_images, axis=1), axis=1)
    #noise_std = np.empty(len(input_images))
    #noise_std[0:len(input_images)] =0
    #for n in tqdm(range(1,50)):
    #    noise_std[n-1] =  bn.nanstd(input_images[0:n+1])
    
    # How does standard deviation of a frame vary with number of frames group averaged together?
    # And how does the difference in pixel intensity between frames vary with number of frames group averaged together?
    #curtailed length of output array by a factor of 4 to speed this up.
    noise_std_groupmean = np.empty(len(input_images)/4)
    noise_dif_groupmean = np.empty(len(input_images)/4)
    #print bn.nanmean(input_images[0:1], axis=0)
    #print bn.nanmean(input_images[2:3], axis=0)
    
    #print bn.nanmean(np.subtract(bn.nanmean(input_images[0:1], axis=0),bn.nanmean(input_images[2:3], axis=0)))
    
    for n in tqdm(range(1,len(input_images)/4)) :
        noise_std_groupmean[n] = bn.nanstd(bn.nanmean(input_images[0:n+1], axis=0))
        noise_dif_groupmean[n] = bn.nanmean(np.absolute(np.subtract(bn.nanmean(input_images[0:n+1], axis=0),bn.nanmean(input_images[n+1:2*(n+1)], axis=0))))
    
    plt.subplot(1,2,2)
    
    plt.yscale('log')
    plt.plot(noise_std_groupmean, 'r', label='noise_std_groupmean')
    plt.plot(noise_dif_groupmean, 'g', label='noise_dif_groupmean')
    plt.legend(loc='upper center')
    #Finish up
    
    plt.show()
    print "Finished Testing!"
    quit()

elif args.grouped:
    print 'Group Averaging', input_images, 'with window size', filterwindow
    
    #figure out how big our final array needs to be and pre-populate it for speed.
    im_out = np.empty((len(input_images)/filterwindow,input_images[0].shape[0],input_images[0].shape[1]))
    
    if args.mean:
        print "Calculating means..."
        #loop through the image and compute means/medians/etc...
        for n in tqdm(range(0,len(input_images)/filterwindow)):
            im_out[n]=bn.nanmean(input_images[n*filterwindow:(n+1)*filterwindow], axis=0)
    
    elif args.median:
        print "Calculating medians...",
        #loop through the image and compute means/medians/etc...
        for n in tqdm(range(0,len(input_images)/filterwindow)):
            im_out[n] = np.uint16(bn.nanmedian(input_images[n*filterwindow:(n+1)*filterwindow], axis=0))
    
    im_out = np.uint16(im_out)
    #io.imsave(filename + '.output.tif',im_out)
    print "Finished!"

elif args.running:
    print "Running Averaging", input_images, " with window size", filterwindow
    #im_temp = io.concatenate_images(im_collection)
    if args.mean:
        print "Calculating means..."
        im_out = bn.move_mean(input_images, window=filterwindow, min_count=1, axis=0)
    elif args.median:
        print "Calculating median..."
        im_out = bn.move_median(input_images, window=filterwindow, min_count=1, axis=0)
    im_out = np.float32(im_out)
    #io.imsave(filename + '.output.tif',im_out)

if args.normalise :
    overall_median = bn.nanmedian(im_out, axis=0)
    im_out = np.float32(im_out / overall_median)
    #print "overall_median", overall_median

if args.manyfiles  :
    print len(im_out)
    print args.manyfiles
    filenameprefix = os.path.splitext(filename)[0]
    os.makedirs('_'+str(filenameprefix)+str(args.manyfiles)+'files')
    for n in tqdm(range(0,len(im_out)/args.manyfiles)):
        io.imsave('_'+filenameprefix+str(args.manyfiles)+'files/'+ filename + '.' + str(filterwindow) + '.output.' + str(n) +'.tif',im_out[n*args.manyfiles:(n+1)*args.manyfiles])
else :
    io.imsave(filename + '.' + str(filterwindow) + '.output.tif',im_out)
    print 'Saved', filename + '.' + str(filterwindow) + '.output.tif'

#Notes for potential parallelizing this
#from joblib import Parallel, delayed #easy parallel processing
# Under Windows, it is important to protect the main loop of code to avoid recursive spawning of subprocesses when using joblib.
# No code should run outside of the "if __name__ == '__main__'" blocks, only imports and definitions.
    #im_out = np.empty((len(input_images)/filterwindow,input_images[0].shape[0],input_images[0].shape[1]))
    #Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    #Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    #for n in tqdm(range(0,len(input_images)/filterwindow)):
    #im_out[n] = Parallel(n_jobs=2) bn.delayed(nanmean)(input_images[n*filterwindow:(n+1)*filterwindow], axis=0) for n in range(0,len(input_images)/filterwindow)