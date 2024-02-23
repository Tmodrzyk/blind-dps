# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:55:58 2021

@author: TL

Non blind deconvolution methods).

"""
root_cluster = '/home/leuliet/Code_Leuliet/intratofpet/'
root_home = '/Users/tleuliet/Documents/Python/Code/Code_Leuliet/intratofpet/'
from pickle import load
import sys
sys.path.append(root_cluster + 'database/')
sys.path.append(root_home + 'database/')


import numpy as np
from skimage.restoration import richardson_lucy, wiener, unsupervised_wiener
import warnings
from obj import phantom
from methods import f_step

def Wiener(image, psf, reg=None, threshold=0):
    """
    Deconvolution with Wiener filter for 2D images.
    See documentation of skimage.restoration from which the function is based.

    Parameters
    ----------
    image : numpy.ndarray
        The input degraded image.
    psf : numpy.ndarray
        Known PSF.
    reg : float, optional
        Regularization parameter. Default is None, in which case unsupervised Wiener 
        method is used to find the best hyperparameters.
    threshold : float, optional
        Values below threshold are put to threshold. In most cases one should keep
        the default value of 0.

    Returns
    ----------
    deconv : numpy.ndarray
        Deconvolved image

    """  
    
    if isinstance(image, phantom):
        image = image.array

    assert isinstance(image, np.ndarray), "image should be phantom of numpy.ndarray"
    assert isinstance(psf, np.ndarray), "psf should be numpy.ndarray"

    assert image.shape == psf.shape, "Only works for same shape" #IMPROVE FEATURE HERE
    assert len(image.shape) == 2, "Only works for 2D images" #IMPROVE FEATURE HERE

    # im_shape = image.shape
    # n = image.shape[0] #assume square image here
    # center = n // 2 #assume square image
    # limit_psf = psf.shape[0] // 2 #assume square psf

    # def augment_matrix(matrix): 
    #     aug_mat = np.zeros(im_shape)
    #     aug_mat[center-limit_psf:center+limit_psf, center-limit_psf:center+limit_psf, :] = matrix
    #     return aug_mat

    if reg is None:
        deconv, _ = unsupervised_wiener(image, psf, clip=False)
    else:
        assert reg >= 0, "reg should be non negative"
        deconv = wiener(image, psf, balance=reg, clip=False)

    deconv[deconv<=threshold] = 0
    deconv = np.reshape(deconv, image.shape)
    
    return deconv



def RL(image, psf, n_iter, implementation='own', threshold=1e-15):
    """
    Deconvolution with Richardson-Lucy algorithm.
    See documentation of skimage.restoration from which the function is based.

    Parameters
    ----------
    image : numpy.ndarray
        The input degraded image.
    psf : numpy.ndarray
        Known PSF.
    n_iter : int
        Number of iterations.
    implementation : str, optional
        'own' or 'skimage' to choose which implementation to use for RL algorithm.
    threshold : float, optional
        Value below which results are set to zero in order to avoid division by small 
        numbers. Default is 1e-15.


    Returns
    ----------
    deconv : numpy.ndarray
        Deconvolved image
    PSNR : list
        list of PSNR values at each iteration if test is True
    CRC : list
        list of CRC values at each iteration if test is True
    KL : list
        list of KL distance values at each iteration if test is True
    TV : list
        list of TV values at each iteration if test is True
    cost : list
        list of cost function at each iteration if test is True
    """  

    
    if isinstance(image, phantom):
        image = image.array


    assert isinstance(image, np.ndarray), "image should be phantom of numpy.ndarray"
    assert isinstance(psf, np.ndarray), "psf should be PSF of numpy.ndarray"

    # if test:
    #     assert implementation == 'own', "test avalailable only for own implementation"
    #     assert isinstance(gt, phantom), "GT must be phantom if test is True"

    #     PSNR_values = []
    #     CRC_values = []
    #     KL_values = []
    #     TV_values = []
    #     cost_values = []


    if implementation == 'skimage':
        warnings.warn('Might be better to choose own implementation since skimage \
                        does not use threshold anymore')
        deconv = richardson_lucy(image, psf, iterations=n_iter, clip=False)
    else:
        assert implementation == 'own', "implementation should be non 'own' or 'skimage'"
        deconv = np.ones(image.shape)
        for _ in range(n_iter):
            deconv = f_step(deconv, psf, image, threshold=threshold)

    #         if test:
    #             PSNR_values.append(PSNR(gt, deconv))
    #             CRC_values.append(mean_CRC(gt, deconv))
    #             KL_val = KL(image, fast_conv(deconv, psf))
    #             KL_values.append(KL_val)
    #             TV_values.append(TV(deconv))
    #             cost_values.append(KL_val)

    # if test:
    #     return deconv, PSNR_values, CRC_values, KL_values, TV_values, cost_values
    # else:
    #     return deconv
    return deconv



def RL_TV(image, psf, n_iter, reg, TV_iter=50, threshold=1e-15):
    """
    Deconvolution with Richardson-Lucy algorithm with TV denoising performed at each step.

    Parameters
    ----------
    image : numpy.ndarray
        The input degraded image.
    psf : numpy.ndarray
        Known PSF.
    n_iter : int
        Number of iterations of RL.
    reg : float
        Regularization parameter
    TV_iter : int, optional
        Number of iterations for TV denoising. Default is 50.
    threshold : float, optional
        Value below which results are set to zero in order to avoid division by small 
        numbers. Default is 1e-15.

    Returns
    ----------
    deconv : numpy.ndarray
        Deconvolved image
    PSNR : list
        list of PSNR values at each iteration if test is True
    CRC : list
        list of CRC values at each iteration if test is True
    KL : list
        list of KL distance values at each iteration if test is True
    TV : list
        list of TV values at each iteration if test is True
    cost : list
        list of cost function at each iteration if test is True

    """  
    
    if isinstance(image, phantom):
        image = image.array

    assert isinstance(image, np.ndarray), "image should be phantom of numpy.ndarray"
    assert isinstance(psf, np.ndarray), "psf should be PSF of numpy.ndarray"

    deconv = np.ones(image.shape)
    for _ in range(n_iter):
        deconv = f_step(deconv, psf, image, threshold=threshold)
        deconv = TV_denoise(deconv, reg=reg, n_iter=TV_iter)

    return deconv


class Deconv:
    """
    Non blind deconvolution.

    Attributes
    ----------
    image : numpy.ndarray
        Array of the estimated image.
    psf : numpy.ndarray
        Array of the estimated PSF.
    blurred : numpy.ndarray
        Considered input blurred image.

    """
    def __init__(self, psf, blurred):
        """
        Parameters
        ----------
        psf : numpy.ndarray or PSF object
            PSF of the system
        blurred : numpy.ndarray, optional
            Convolved data. If None, projections should be given to consider the FBP as the convolved data.
        projections : Sinogram object, optional
            Projections of the convolved data. If provided, FBP is computed and taken as the input of 
            the deconvolution algorithm.
        """

        assert isinstance(psf, np.ndarray), "PSF should be PSF object or numpy.ndarray"
        self.psf = psf
        self.blurred = blurred
        self.image = np.ones_like(self.blurred)

    def Wiener_deconv(self, reg=None, threshold=0):
        """
        Deconvolution with Wiener filter for 2D images.
        See documentation of skimage.restoration from which the function is based.

        Parameters
        ----------
        reg : float, optional
            Regularization parameter. Default is None, in which case unsupervised Wiener 
            method is used to find the best hyperparameters.
        threshold : float, optional
            Values below threshold are put to threshold. In most cases one should keep
            the default value of 0.
            """
        self.image = Wiener(self.blurred, self.psf, reg=reg, threshold=threshold)


    def RL_deconv(self, n_iter, imp='own', thres=1e-15):
        """
        Deconvolution with Richardson-Lucy algorithm.
        See documentation of skimage.restoration from which the function is based.

        Parameters
        ----------
        n_iter : int
            Number of iterations.
        imp : str, optional
            'own' or 'skimage' to choose which implementation to use for RL algorithm.
        thres : float, optional
            Value below which results are set to zero in order to avoid division by small 
            numbers. Default is 1e-15.
  
        """
        
        self.image = RL(self.blurred, self.psf, n_iter, implementation=imp, threshold=thres)


    def RL_TV_deconv(self, n_iter, reg, TV_iter=50, thres=1e-15):
        """
        Deconvolution with Richardson-Lucy algorithm with TV denoising performed at each step.

        Parameters
        ----------
        n_iter : int
            Number of iterations of RL.
        reg : float
            Regularization parameter
        TV_iter : int, optional
            Number of iterations for TV denoising. Default is 50.
        thres : float, optional
            Value below which results are set to zero in order to avoid division by small 
            numbers. Default is 1e-15.
        """

        self.image = RL_TV(self.blurred, self.psf, n_iter, reg, TV_iter=TV_iter, threshold=thres)

    def TV_denoise(self, reg, TV_iter=50):
        """
        Performs TV denoising on the restored image.
        
        Parameters
        ----------
        reg : float
            Regularization parameter
        TV_iter : int, optional
            Number of iterations for TV denoising. Default is 50.
        """
        self.image = TV_denoise(self.image, reg, TV_iter)
