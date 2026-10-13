python fetusnet.py \
  --mode=script_mean_template \
  --ds.sys="/media/yusuf/HDD 4TB/" \
  --ds.root="DATA/" \
  --ds.dataframe="sinMaternitat__fold0" \
  --ds.index_order="C" \
  --device="cuda" \
  --script.spatial_prior.exp_dir="runs/2026-05-19/kendall6/" \
  --script.mean_template.validation=true