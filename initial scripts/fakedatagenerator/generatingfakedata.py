import numpy as np
from skimage import io
from random import uniform
import pylab as py
from matplotlib import pyplot as plt # for plotting
#from astropy.modeling import models


def makeGaussian(boxsize, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    boxsize is the length of a side of the square
    fwhm is full-width-half-maximum.
    """

    x = np.arange(0, boxsize, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = boxsize // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    

def RandomWalk():
    nsteps = 10000
    steplength = 1
    theta = 2 * np.pi * np.random.rand(1, nsteps - 1)
    xy = np.dstack((steplength*np.cos(theta), steplength*np.sin(theta)))
    walk1 =  np.cumsum(xy, axis=1)
    walk2 = walk1[0]
    return walk2
        
    
output = np.random.normal(300,5,[10000,64,64]) #mean stdev, boxsize of array
output += (1000-300)*makeGaussian(64,64,None) #add some intensity variation for illumination profile

#output += 0.1*(1000-300)*makeGaussian(64,5,[uniform(0,64),uniform(0,64)]) 
#output += 500*makeGaussian(64,64,None)


testwalk=RandomWalk()
counter = 0
for xy in testwalk:
    #print xy
    output[counter,:,:] *= (0.1*makeGaussian(64,5,xy+[32,32])+1)
    counter +=1
output = np.uint16(output)# convert to 16 bit
io.imsave('output.tif',output)

#plt.imshow(output[4], cmap='gray')
#plt.show()



