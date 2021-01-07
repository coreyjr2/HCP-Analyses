#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:34:45 2020

@author: cjrichier
"""

import dicom2nifti

dicom2nifti.convert_directory('/Users/cjrichier/Downloads/DICOM/', '/Users/cjrichier/Desktop/Taylor/')


import nibabel as nb


'''Here is the first image'''
axial = nb.load('/Users/cjrichier/Desktop/Taylor/5_ax_dwi.nii.gz')

axial.shape
axial.get_fdata
axial.affine
print(axial.header)
print(axial.header['sform_code'])
#RAS was aligned to another scan

#Split image into T1 and T2
axial_T1 = axial.slicer[:,:,:,0]
axial_T1.shape

axial_T2 = axial.slicer[:,:,:,1]
axial_T2.shape

#plot the pictures of the T1 and T2
from nilearn import plotting
plotting.plot_img(axial_1)
plotting.plot_img(axial_2)



img2 = nb.load('/Users/cjrichier/Desktop/Taylor/4_sag_t1_se.nii.gz')
img2.shape
img2.get_fdata
img2.affine
print(img2.header)
print(img2.header['sform_code'])
#RAS was aligned to another scan

plotting.plot_img(img2)


img3 = nb.load('/Users/cjrichier/Desktop/Taylor/501_exponential_apparent_diffusion_coefficient.nii.gz')
img3.shape
plotting.plot_img(img3)