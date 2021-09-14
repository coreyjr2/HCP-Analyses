
import nilearn as nl
import numpy as np
import nibabel as nib
import pandas as pd
import sklearn
import datetime as dt
import os
import platform
import json
import hashlib
import sys
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker
from nilearn import plotting

sep = os.path.sep
source_path = os.path.abspath(os.getcwd()) + sep

data_path_template = 'S:\\HCP\\HCP_1200\\{subject}\\MNINonLinear\\Results\\{session}_{run}\\{session}_{run}.nii.gz'.format(subject = '100206', session = 'tfMRI_MOTOR', run= 'LR')
# raw_timeseries = nib.load(data_path_template)


# atlas_ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
# atlas_filename_ho = atlas_ho.maps

# masker_ho = NiftiLabelsMasker(labels_img=atlas_filename_ho, standardize=True)
# masked_timeseries_ho = []
# masked_timeseries_ho = masker_ho.fit_transform(raw_timeseries)


# atlas_msdl = datasets.fetch_atlas_msdl()
# atlas_filename_msdl = atlas_msdl.maps

# masker_msdl = NiftiMapsMasker(maps_img=atlas_filename_msdl, standardize=True, memory='nilearn_cache')
# masked_timeseries_msdl = []
# masked_timeseries_msdl = masker_msdl.fit_transform(raw_timeseries)


# atas_glasser_01_filename = 'S:\\Code\\Parcellation\\MMP 1.0 MNI projections\\MMP_in_MNI_corr.nii.gz'
# atlas_glasser_01 = nib.load(atas_glasser_01_filename)

# masker_glasser_01 = NiftiLabelsMasker(labels_img=atas_glasser_01_filename, standardize=True)
# masked_timeseries_glasser_01 = []
# masked_timeseries_glasser_01 = masker_glasser_01.fit_transform(raw_timeseries)
# masked_timeseries_glasser_01.shape


# atas_glasser_02_filename = 'S:\\Code\\Parcellation\\HCP-MMP1\\HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz'
# atlas_glasser_02 = nib.load(atas_glasser_02_filename)

# masker_glasser_02 = NiftiLabelsMasker(labels_img=atas_glasser_02_filename, standardize=True)
# masked_timeseries_glasser_02 = []
# masked_timeseries_glasser_02 = masker_glasser_02.fit_transform(raw_timeseries)
# masked_timeseries_glasser_02.shape



# atas_glasser_03_filename = 'S:\\Code\\Parcellation\\HCP-MMP1\\HCP-MMP1_on_MNI152_ICBM2009a_nlin_hd.nii.gz'
# atlas_glasser_03 = nib.load(atas_glasser_03_filename)

# masker_glasser_03 = NiftiLabelsMasker(labels_img=atas_glasser_03_filename, standardize=True)
# masked_timeseries_glasser_03 = []
# masked_timeseries_glasser_03 = masker_glasser_03.fit_transform(raw_timeseries)
# masked_timeseries_glasser_03.shape

# atas_hmat_filename = 'S:\\Code\\Parcellation\\HMAT_website\\HCP-MMP1_on_MNI152_ICBM2009a_nlin_hd.nii.gz'
# atlas_hmat = nib.load(atas_hmat_filename)

# masker_hmat = NiftiLabelsMasker(labels_img=atas_hmat_filename, standardize=True)
# masked_timeseries_hmat = []
# masked_timeseries_hmat = masker_hmat.fit_transform(raw_timeseries)
# masked_timeseries_hmat.shape

def parcellate_timeseries(nifty_file, atlas_name, confounds=None):
  # Other atlases in MNI found here: https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
  raw_timeseries = nib.load(nifty_file)
  if atlas_name=='harvard_oxford':
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
    atlas_filename = atlas.maps
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
  elif atlas_name == 'msdl':
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas.maps
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache')
  elif atlas_name == 'mni_glasser':
    atas_glasser_01_filename = source_path + 'MMP_in_MNI_corr.nii.gz'
    masker = NiftiLabelsMasker(labels_img=atas_glasser_01_filename, standardize=True)
  elif 'yeo' in atlas_name:
    yeo = datasets.fetch_atlas_yeo_2011()
    if atlas_name == 'yeo_7_thin':
      masker = NiftiLabelsMasker(labels_img=yeo['thin_7'], standardize=True,memory='nilearn_cache')
    elif atlas_name == 'yeo_7_thick':
      masker = NiftiLabelsMasker(labels_img=yeo['thick_7'], standardize=True,memory='nilearn_cache')
    elif atlas_name == 'yeo_17_thin':
      masker = NiftiLabelsMasker(labels_img=yeo['thin_17'], standardize=True,memory='nilearn_cache')
    elif atlas_name == 'yeo_17_thick':
      masker = NiftiLabelsMasker(labels_img=yeo['thick_17'], standardize=True,memory='nilearn_cache')
  #Transform the motor task imaging data with the masker and check the shape
  masked_timeseries = []
  if confounds is not None:
    masked_timeseries = masker.fit_transform(raw_timeseries, counfounds = confounds)
  else:
    masked_timeseries = masker.fit_transform(raw_timeseries)
  return masked_timeseries

out = {}
for atlasname in ['harvard_oxford','msdl','mni_glasser','yeo_7_thin','yeo_7_thick','yeo_17_thin','yeo_17_thick']:
  out[atlasname] = parcellate_timeseries(data_path_template, atlasname)
  print(atlasname, out[atlasname].shape)