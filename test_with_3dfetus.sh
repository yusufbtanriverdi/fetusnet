python fetusnet.py \
  --mode=test \
  --checkpoint="runs/2026-05-19/kendall6" \
  --validation.use_model="last" \
  --validation.filter_3dFetusTest=true \
  --ds.sys="/media/yusuf/HDD 4TB/" \
  --ds.root="DATA_gtpp_C_sitk/" \
  --ds.dataframe="3dfetus_test_fold0" \
  --ds.index_order="F" \
  --device="cuda" \
  --test_patients="[0]" \
  --validation.detector="argmax" \
  --validation.radius_eval=100 \
  --validation.radius_num=100 \
  --validation.save_outputs=True \
  --validation.save_targets=True \
  --validation.show_figures=False

# python fetusnet.py \
#   --mode=test \
#   --checkpoint="runs/2026-05-20/kendall6_f1" \
#   --validation.use_model="last" \
#   --validation.filter_3dFetusTest=true \
#   --ds.sys="/media/yusuf/HDD 4TB/" \
#   --ds.root="DATA_gtpp_fin/" \
#   --ds.dataframe="3dfetus_test_fold1" \
#   --ds.index_order="F" \
#   --device="cuda" \
#   --test_patients="[0]" \
#   --validation.detector="argmax" \
#   --validation.radius_eval=100 \
#   --validation.radius_num=100 \
#   --validation.save_outputs=True \
#   --validation.save_targets=True \
#   --validation.show_figures=False

# python fetusnet.py \
#   --mode=test \
#   --checkpoint="runs/2026-05-20/kendall6_f2" \
#   --validation.use_model="last" \
#   --validation.filter_3dFetusTest=true \
#   --ds.sys="/media/yusuf/HDD 4TB/" \
#   --ds.root="DATA_gtpp_fin/" \
#   --ds.dataframe="3dfetus_test_fold2" \
#   --ds.index_order="F" \
#   --device="cuda" \
#   --test_patients="[0]" \
#   --validation.detector="argmax" \
#   --validation.radius_eval=100 \
#   --validation.radius_num=100 \
#   --validation.save_outputs=True \
#   --validation.save_targets=True \
#   --validation.show_figures=False

# python fetusnet.py \
#   --mode=test \
#   --checkpoint="runs/2026-05-20/kendall6_f3" \
#   --validation.use_model="last" \
#   --validation.filter_3dFetusTest=true \
#   --ds.sys="/media/yusuf/HDD 4TB/" \
#   --ds.root="DATA_gtpp_fin/" \
#   --ds.dataframe="3dfetus_test_fold3" \
#   --ds.index_order="F" \
#   --device="cuda" \
#   --test_patients="[0]" \
#   --validation.detector="argmax" \
#   --validation.radius_eval=100 \
#   --validation.radius_num=100 \
#   --validation.save_outputs=True \
#   --validation.save_targets=True \
#   --validation.show_figures=False

# python fetusnet.py \
#   --mode=test \
#   --checkpoint="runs/2026-05-28/kendall6_norot" \
#   --validation.use_model="last" \
#   --validation.filter_3dFetusTest=true \
#   --ds.sys="/media/yusuf/HDD 4TB/" \
#   --ds.root="DATA_gtpp_fin/" \
#   --ds.dataframe="3dfetus_test_fold0" \
#   --ds.index_order="F" \
#   --device="cuda" \
#   --test_patients="[0]" \
#   --validation.detector="argmax" \
#   --validation.radius_eval=100 \
#   --validation.radius_num=100 \
#   --validation.save_outputs=True \
#   --validation.save_targets=True \
#   --validation.show_figures=False

# python fetusnet.py \
#   --mode=test \
#   --checkpoint="runs/2026-06-01/kendall6_norot_f3" \
#   --validation.use_model="last" \
#   --validation.filter_3dFetusTest=true \
#   --ds.sys="/media/yusuf/HDD 4TB/" \
#   --ds.root="DATA_gtpp_fin/" \
#   --ds.dataframe="3dfetus_test_fold3" \
#   --ds.index_order="F" \
#   --device="cuda" \
#   --test_patients="[0]" \
#   --validation.detector="argmax" \
#   --validation.radius_eval=100 \
#   --validation.radius_num=100 \
#   --validation.save_outputs=True \
#   --validation.save_targets=True \
#   --validation.show_figures=False