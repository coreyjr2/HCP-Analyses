#!/bin/bash
n_features=$1
version=$2
split=$3

python3 /data/hx-hx1/kbaacke/Code/HCP-Analyses/hcp_cluster_predGen.py \
  -n_features $n_features \
  -version $version \
  -split $split