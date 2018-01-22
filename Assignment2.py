#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:38:14 2017

@author: jamesdawson
"""

from scipy.misc import imread,imsave
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

###############################################################################
### Reading in the image and converting to greyscale ##########################
###############################################################################

'''
image = imread('Henrik.png')

new_im=  np.zeros([len(image[:,0]),len(image[0,:])])

def average_pixel(image):
        for i in range(len(image[:,0])):
                for j in range(len(image[0,:])):
                        new_im[j,i] = np.mean(image[j,i])
                        
grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
# get row number
for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = np.mean(image[rownum][colnum])
                
plt.imshow(grey, cmap = plt.cm.Greys_r)
plt.show()

imsave('Henrik2.png', grey)

'''

im = imread('Henrik2.png')
im = im/255.

###############################################################################
###############################################################################


###############################################################################
### Noise models to add to images #############################################
###############################################################################

def add_gaussian_noise(im,prop,varSigma):
        N = int(np.round(np.prod(im.shape)*prop))
        index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
        e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
        im2 = np.copy(im)
        im2[index] += e[index]
        return im2
def add_saltnpeppar_noise(im,prop):
        N = int(np.round(np.prod(im.shape)*prop))
        index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
        im2 = np.copy(im)
        im2[index] = 1-im2[index]
        return im2

###############################################################################
###############################################################################


###############################################################################        
### Altering the images by adding noise #######################################
###############################################################################

im[np.where(im<0.5)] = -1.
im[np.where(im>0.5)] = 1.
# proportion of pixels to alter
prop = 0.1
varSigma = 0.1
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(im,cmap='gray')
im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(132)
ax2.imshow(im2,cmap='gray')
im3 = add_saltnpeppar_noise(im,prop)
ax3 = fig.add_subplot(133)
ax3.imshow(im3,cmap='gray')



###############################################################################
###############################################################################


###############################################################################
### Defining the nearest neighbours ###########################################
###############################################################################

def neighbours(i,j,M,N,size=4):
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        print('Not yet implemented\n')
        return -1

###############################################################################
###############################################################################

        
###############################################################################
# Ising model for determining the latent variables of the image ###############
###############################################################################


def energy_function(im,y,n,h,beta,nu,x_j):
        sum_1 = 0.
        sum_3 = 0.
        for k in range(0,len(n)):
            sum_1 += im[n[k]]
            sum_3 += im[n[k]]*y[n[k]]
        term_1 = sum_1*h
        term_2 = sum_1*-beta*x_j
        term_3 = -nu*sum_3
        return term_1+term_2+term_3
                
def ICM(image,h,alpha,beta):    
        y = np.copy(image)
        for i in range(len(image[:,0])):
                for j in range(len(image[0,:])):
                        n = neighbours(i,j,len(y[:,0]),len(y[0,:]),size=4)                               
                        plus = energy_function(image,y,n,h,alpha,beta,1.0)
                        minus = energy_function(image,y,n,h,alpha,beta,-1.0)
                        if plus > minus:
                                y[i,j] = 1.
                        else:
                                y[i,j] = -1.
        d_copy = np.copy(y)
        d_copy[np.where(y==-1.0)] = 1.0
        d_copy[np.where(y==1.0)] = -1.0
        
        return d_copy
 
### Unhash below to run this model ############################################

T = 50
h = 0
image_2 = np.copy(im2)
#fig,axs = plt.subplots(3,3, figsize=(15, 15), facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = 0.1, wspace=0.)
#axs = axs.ravel()
for i in range(T):
#        print(i)                                       
        image_2 = ICM(image_2,h,3,1)
#        axs[i].imshow(image_2,cmap='gray')
#plt.savefig('denoising'+str(i))

plt.figure()
plt.subplot(131)
plt.imshow(image_2,cmap='gray')


                
###############################################################################
###############################################################################



###############################################################################
# Gibbs sampling model for determining latent variables of the image ##########
###############################################################################

def Gibbs_ICM(im,h,alpha,beta):
    shape=np.shape(im)
    y = np.copy(im)
    M = shape[0]
    N = shape[1]
    for i in range(0,M):
        for j in range(0,N):
            n = neighbours(i,j,len(y[:,0]),len(y[0,:]),size=4) 
            E_neg = energy_function(im,y,n,h,alpha,beta,-1.0)
            E_pos = energy_function(im,y,n,h,alpha,beta,1.0)
            posterior = E_pos/(E_pos+E_neg)
            t = np.random.random()
            if posterior>t:
                y[i,j] = 1.0
            else:
                y[i,j] = -1.0
    return y

### Unhash below to run this model ############################################

T = 50
h = 0
image_2 = np.copy(im3)
#fig,axs = plt.subplots(3,3, figsize=(15, 15), facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = 0.1, wspace=0.)
#axs = axs.ravel()
for i in range(T):
#        print(i)                                       
        image_2 = Gibbs_ICM(image_2,h,3,1.)
#        axs[i].imshow(image_2,cmap='gray')
#plt.savefig('denoising_gibbs'+str(i))

#plt.figure()
plt.subplot(132)
plt.imshow(image_2,cmap='gray')


###############################################################################
###############################################################################

np.random.seed(42)

def Gibbs_ICM_rnd(im,h,alpha,beta):
    shape=np.shape(im)
    y = np.copy(im)
    M = shape[0]
    N = shape[1]
    r = list(range(M))
    np.random.shuffle(r)
    for i in r:
            s = list(range(N))
            np.random.shuffle(s)
            for j in s:                   
                    n = neighbours(i,j,M,N,size=4) 
                    E_neg = energy_function(im,y,n,h,alpha,beta,-1.0)
                    E_pos = energy_function(im,y,n,h,alpha,beta,1.0)
                    posterior = E_pos/(E_pos+E_neg)
                    t = np.random.uniform(0,1)
                    if posterior>t:
                        y[i,j] = 1
                
                    else:
                        y[i,j] = -1
    return y

### Unhash below to run this model ###########################################
T = 50
h = 0
image_2 = np.copy(im3)
#fig,axs = plt.subplots(3,3, figsize=(15, 15), facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = 0.1, wspace=0.)
#axs = axs.ravel()
for i in range(T):
        print(i)                                     
        image_2 = Gibbs_ICM_rnd(image_2,h,3,1)
#        axs[i].imshow(image_2,cmap='gray')
   
          
#axs[2].imshow(image_2,cmap='gray') 
#plt.savefig('denoising_gibbs_rnd1')

#plt.figure()
plt.subplot(133)
plt.imshow(image_2,cmap='gray')
plt.savefig('long')
