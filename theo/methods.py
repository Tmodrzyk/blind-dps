# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:55:58 2021

@author: TL

Methods for algorithms.py

"""

import numpy as np
from scipy.signal import convolve
from skimage.restoration import denoise_tv_chambolle


def fast_conv(a, b, mode='same'):
    '''
    Same function as scipy.signal.convolve but default mode is changed to 'same' not to specify
    it every time.
    
    Parameters
    ----------
    a : numpy.ndrray
        input 1 for the convolution
    b : numpy.ndarray
        input 2 for the convolution
    mode : str
        'same' (default), 'full' or 'valid'

        See documentation from scipy.signal.convole.
    
    Returns
    ------
    conv : numpy.ndarray
          Convolution of a and b.
    '''
    conv = convolve(a, b, mode=mode) 
    #force values to be non negative if both inputs are 
    if np.all(a>=0) and np.all(b>=0):
        conv[conv<0] = 0
    return conv


def f_step(f_k, g, data, threshold=1e-15):
    """
    One iteration of Richardson-Lucy algo for the image.

    Parameters
    ----------
    f_k : numpy.ndarray
        Current estimate for the image.
    g : numpy.ndarray
        Estimate for the true PSF.
    data : numpy.ndarray
        Data from the convolution of image and PSF. 
    threshold : float, optional 
        Threshold for which the pixel/voxel will be set to 0 if value is lower, in order
        to prevent division by very low values that might make the algorithm diverge.

    Returns
    ----------
    f_next : numpy.ndarray
        Next iteration for the image.
        
    """  
    s = fast_conv(np.ones(f_k.shape), np.flip(g)) #weighting parameter

    conv = fast_conv(f_k, g)
    quot = np.divide(data, conv, out=np.zeros_like(data), where=conv!=0)

    conv = fast_conv(quot, np.flip(g))
    f_next = (f_k / s) * conv

    f_next[f_next<=threshold] = 0 #to avoid division by small numbers
    return f_next

def gradient(u, to_add='last') :
    """
    Compute the gradient of an array u
    
    Parameters
    ----------
    u : numpy array 
        matrix one wants the gradient of (dim 3)
    to_add : str
        'last' or 'first', column/row to add to compute finite differences in order to 
        retrieve the same shape as the input.
        When using gradient to compute divergence, should set to 'first', otherwise
        keep 'last'
    
    Returns
    ------
    g : numpy array 
        gradient of u of shape (3, shape_u)
    """

    assert len(u.shape)==3, "function gradient adapted for dim 3" 

    #fix rows/columns to add for computation, could be optimized
    if to_add == 'last':
        app1 = u[-1,:,:].reshape(1,u.shape[1],u.shape[2])
        app2 = u[:,-1,:].reshape(u.shape[0],1,u.shape[2])
        app3 = u[:,:,-1].reshape(u.shape[0],u.shape[1],1)
        prep1, prep2, prep3 = np._NoValue, np._NoValue, np._NoValue
    elif to_add == 'first':
        prep1 = u[0,:,:].reshape(1,u.shape[1],u.shape[2])
        prep2 = u[:,0,:].reshape(u.shape[0],1,u.shape[2])
        prep3 = u[:,:,0].reshape(u.shape[0],u.shape[1],1)
        app1, app2, app3 = np._NoValue, np._NoValue, np._NoValue

    grad = np.zeros((3, u.shape[0], u.shape[1], u.shape[2]), dtype = 'float32')
    grad[0] = np.diff(u, 1, axis=0, append=app1, prepend=prep1)     
    grad[1] = np.diff(u, 1, axis=1, append=app2, prepend=prep2)
    grad[2] = np.diff(u, 1, axis=2, append=app3, prepend=prep3)

    return grad  

def divergence(g) :
    '''
    Compute the divergence of the vector field u
    
    Parameters
    ----------
     g : numpy array
        vector field of dimension (3, dim_x, dim_y, dim_z)
    
    Return
    ------
    div : numpy array
          array of dimension (dim_x, dim_y, dim_z)
    '''
    assert len(g.shape) == 4, "input for divergence should be 4 dimensional"



    grad_x = gradient(g[0], to_add='first')[0]
    grad_y = gradient(g[1], to_add='first')[1]
    grad_z = gradient(g[2], to_add='first')[2]

    return grad_x + grad_y + grad_z


def TV_denoise(f, reg, n_iter, algo='L2-TV'):
    """
    Total variation denoising based on Chambolle Pock algorithm or Chambolle algorithm from 2004.
    Minimizaton of L2-TV with Chambolle algorithm, minimizaton of KL-TV with Chambolle Pock algorithm.

    Implementation is based on skimage.restoration library for Chambolle algorithm.
    Note in that case that default epsilon stopping criterion is considered and has 0.0002 for value.

    For CP, this is the original formulation of the Chambolle Pock algorithm (no preconditioning)
    as it is presented in Sidky2012 algorithm 5 with A = Id.
    
    Parameters
    ----------
    f : numpy.ndrray
        Image to denoise.
    reg : float
        Regularization parameter.
    n_iter : int
        Number of iterations
    algo : str, optional
        Algorithm to use for TV denoising.
        If 'KL-TV', Chambolle Pock algorithm as presented in Algorithm 5 in Sidky2012 is used,
        where the data-fidelity term is the Kullback-Leibler divergence.
        If 'L2-TV', Chambolle algorithm as presented in Chambolle2004 is used, where the data-
        fidelity term is the L2 norm.
    
    Returns
    ------
    denoised : numpy.ndarray
          Denoised image.
    """    

    assert isinstance(f, np.ndarray), "image should be numpy.ndarray"
    
    assert len(f.shape) == 2, "needs to be adapted for 2D arrays!"
    
    if algo == 'KL-TV':
        L = 1 #norm of (1,grad)
        tau = 1 / L
        sigma = 1 / L
        theta = 1   

        n = 0

        u = np.zeros_like(f)
        u_bar = np.zeros_like(f)
        p = np.zeros_like(f)
        q = np.zeros_like(gradient(f))

        while n < n_iter:
            p = 0.5 * (1 + p + sigma*u_bar - np.sqrt((p+sigma*u_bar-1)**2 + 4*sigma*f))

            qgrad = q + sigma*gradient(u_bar) 
            q = reg * qgrad / max(np.linalg.norm(qgrad), reg)
                    
            u_int = u.copy()
            u = u - tau*p + tau*divergence(q)

            u_bar = u + theta*(u-u_int)
            del u_int
            
            u_bar[u_bar<=0] = 0 

            n += 1

        return u_bar

    else:
        assert algo == 'L2-TV', "algo should be KL-TV or L2-TV"
        denoised = denoise_tv_chambolle(f, weight=reg, n_iter_max=n_iter)

        return denoised