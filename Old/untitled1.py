#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 17:44:00 2020

@author: cjrichier
"""

'''Load in some relevant libraries'''

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import json

import nibabel as nib

def load_image(path):
    # load an img file
    return nib.load(path)

def get_TR(img):
    # retrieve TR data
    return img.header.get_zooms()[-1]

def get_slices(img):
    # retrieve number of slices
    return img.shape[2]
  
def get_header(img):
    # print the full header
    return(img.header)

'''load in a single subject's data'''
path=HCP_DIR = '/Volumes/Byrgenwerth/Datasets/ds000030-download/sub-10228/func/sub-10228_task-rest_bold.nii.gz'
img = load_image(path)

'''let's check out some information about this subject's data'''
TR = get_TR(img)
slices = get_slices(img)
print('TR: {}'.format(TR))
print('# of slices: {}'.format(slices))
img.shape


'''Take a peek at the metadata'''
header=get_header(img)
print(header)


'''Let's do a feature extraction now'''

from nilearn import input_data
from nilearn import datasets
from nilearn import plotting
from nilearn.plotting import plot_prob_atlas, plot_roi, plot_matrix

from nilearn.decomposition import CanICA 
from nilearn import image
from nilearn.regions import RegionExtractor


from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')



'''Upload the images into your console'''
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

# upload 2 fMRI scans files. 
img_sub_50006=load_image('/Volumes/Byrgenwerth/Datasets/ds000030-download/sub-50006/func/sub-50006_task-rest_bold.nii.gz')
img_sub_10228=load_image('/Volumes/Byrgenwerth/Datasets/ds000030-download/sub-10228/func/sub-10228_task-rest_bold.nii.gz')

'''These are the raw fMRI files, we will need them to get two things. 
First, an understanding of what regions are what.
Secondly, we can extract the time series of these data as well. 
We will use an seed-based correlation analysis (SCA) and 
a independent components analysis (ICA)'''


'''First we will build a masker. Maskers are used 
frequently in neuroimaging data analysis to process raw image filea
to get them to do what we want them to do. More specifically, 
the shape of data in its current state, a 4D series of images,
isn't really im the best shape for building statistical models,
so we will need to do some finagling to get it in the shape we 
need it to be in. Maskers can also filter fMRI data to extract 
only the parts we care about.'''


## craete masker based on the atlas 
## and create a time series of the uploaded image using the masker
def create_mask(atlas_img, fmri_img):
  # generates a mask given img and atlas
  masker=NiftiLabelsMasker(labels_img=atlas_img, standardize=True)
  time_series=masker.fit_transform(fmri_img)
  
  return time_series

# using the correlation measures defined above, 
# we calculate the correaltion matrixes
def calc_correlation_matrix(time_series):
  # given a time series, return a correlation matrix
  return correlation_measure.fit_transform([time_series])[0]

#and we plot,
def plot_cor_matrix(correlation_matrix, title):
  ## plot the correlation matrix

  np.fill_diagonal(correlation_matrix, 0)
  plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                       vmax=0.8, vmin=-0.8, reorder=True)
  plt.title(title)
  plt.show()



'''Now we will load in an atlas to define what region goes where.'''
## import an existing map

# Harvard-Oxford
'''
retirived from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
'''
harvard_dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
harvard_maps = harvard_dataset.maps
harvard_labels = harvard_dataset.labels

## Smith
smith_atlas = datasets.fetch_atlas_smith_2009()
smith_atlas_rs_networks = smith_atlas.rsn10

# plot the ROIs

#Harvard-Oxford is a built-in atals
plot_roi(harvard_maps, title='atlas harvard oxford ROIs')

#For the Smith's atlas, we need to extract the regions from the nifti object
extraction = RegionExtractor(smith_atlas_rs_networks, min_region_size=800,
                             threshold=98, thresholding_strategy='percentile')
extraction.fit()
smith_maps = extraction.regions_img_

#and to plot
plot_prob_atlas(smith_maps, title="Smith rsn regions extracted.")


# An example for correlation matrix using both atlases

img=img_sub_10228 


# Smith
# We take the first out of the 10 rsns
smith_1st_rsn=image.index_img(smith_atlas_rs_networks, 0)

smith_time_series=create_mask(smith_1st_rsn, img)
smith_cor_matrix=calc_correlation_matrix(smith_time_series)
plot_cor_matrix(smith_cor_matrix, 'smith correlation matrix')

# Harvard Oxford
harvard_oxford_time_series=create_mask(harvard_maps, img)
harvard_oxford_cor_matrix=calc_correlation_matrix(harvard_oxford_time_series) 
plot_cor_matrix(harvard_oxford_cor_matrix, 'harvard oxford correlation matrix')











































