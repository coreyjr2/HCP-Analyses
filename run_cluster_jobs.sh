source /usr/local/sge/default/common/settings.sh

dos2unix /data/hx-hx1/kbaacke/Code/HCP-Analyses/hcp_predgen.sh
dos2unix /data/hx-hx1/kbaacke/Code/HCP-Analyses/hcp_cluster_predGen.py
chmod u+x /data/hx-hx1/kbaacke/Code/HCP-Analyses/hcp_predgen.sh

for FEATURES in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 18 19 20 21 22 23 24 25 26 27 28 29 33 34 35 38 39 40 42 43 44 48 50 51 52 54 57 60 62 63 64 67 69 73 75 76 79 81 82 83 84 91 95 97 101 103 110 115 122 125 129 134 141 149 152 156 161 169 179 187 195 208 216 223 242 254 264 275 297 320 340 365 397 414 443 480 522 569 622 676 734 807 876 960 1078 1205 1354 1533 1725 1980 2297 2679 3149 3771 4638 5847 7482 10027 13981 18540 19884; do
  for VERSION in 0 1 2 3 4 5 6 7 8 9; do 
    for SPLIT in 0 1 2 3 4 5 6 7 8 9; do
      qsub -e /data/hx-hx1/kbaacke/SGE_Output/hcp_analysis/ -N _${FEATURES}_${VERSION}_${SPLIT} -l h_vmem=6G /data/hx-hx1/kbaacke/Code/HCP-Analyses/hcp_predgen.sh "$FEATURES" "$VERSION" "$SPLIT"
    done
  done
done

for FEATURES in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 18 19 20 21 22 23 24 25 26 27 28 29 33 34 35 38 39 40 42 43 44 48 50 51 52 54 57 60 62 63 64 67 69 73 75 76 79 81 82 83 84 91 95 97 101 103 110 115 122 125 129 134 141 149 152 156 161 169 179 187 195 208 216 223 242 254 264 275 297 320 340 365 397 414 443 480 522 569 622 676 734 807 876 960 1078 1205 1354 1533 1725 1980 2297 2679 3149 3771 4638 5847 7482 10027 13981 18540 19884; do
  for VERSION in 0 1 2 3 4 5 6 7 8 9; do 
    for SPLIT in 0 1 2 3 4 5 6 7 8 9; do
      qsub -e /data/hx-hx1/kbaacke/SGE_Output/hcp_analysis/ -N RandomPredGen_${FEATURES}_${VERSION}_${SPLIT} -l h_vmem=6.5G /data/hx-hx1/kbaacke/Code/HCP-Analyses/hcp_predgen.sh "$FEATURES" "$VERSION" "$SPLIT"
    done
  done
done


cd /data/hx-hx1/kbaacke/SGE_Output/hcp_analysis
ls -lh

cd /data/hx-hx1/kbaacke/datasets/hcp_analysis_output/8d2513/Accuracies_Temp
ls

for FEATURES in 24 25; do
  for VERSION in 0 1; do 
    for SPLIT in 0 1; do
      qsub -e /data/hx-hx1/kbaacke/SGE_Output/hcp_analysis/ -N f${FEATURES}_v${VERSION}_s${SPLIT} -l h_vmem=6.5G /data/hx-hx1/kbaacke/Code/HCP-Analyses/hcp_predgen.sh "$FEATURES" "$VERSION" "$SPLIT"
    done
  done
done

