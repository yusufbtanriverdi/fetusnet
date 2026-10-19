# python fetusnet.py \
#     --mode=preprocess \
#     --ds.sys="/media/yusuf/HDD 4TB/" \
#     --ds.dataframe="sinMaternitat__fold0" \
#     --preprocessing.save_dir="DATA_path3_F" \
#     --preprocessing.path1=False \
#     --preprocessing.path2=False \
#     --preprocessing.path3=True \
#     --preprocessing.path4=False \
#     --preprocessing.path5=False\
#     --preprocessing.filter.filter_size=3 \
#     --preprocessing.bspline.spacing="[1, 1, 1]" \
#     --preprocessing.bspline.size="[128, 128, 128]" \
#     --write_index_order: "F"

python fetusnet.py \
    --mode=preprocess \
    --ds.sys="/media/yusuf/HDD 4TB/" \
    --ds.dataframe="sinMaternitat__fold0" \
    --preprocessing.save_dir="DATA_path1_C" \
    --preprocessing.path1=True \
    --preprocessing.path2=False \
    --preprocessing.path3=False \
    --preprocessing.path4=False \
    --preprocessing.path5=False\
    --preprocessing.filter.filter_size=3 \
    --preprocessing.bspline.spacing="[1, 1, 1]" \
    --preprocessing.bspline.size="[128, 128, 128]" \
    --write_index_order: "C"

python fetusnet.py \
    --mode=preprocess \
    --ds.sys="/media/yusuf/HDD 4TB/" \
    --ds.dataframe="sinMaternitat__fold0" \
    --preprocessing.save_dir="DATA_path2_pippi" \
    --preprocessing.path1=False \
    --preprocessing.path2=True \
    --preprocessing.path3=False \
    --preprocessing.path4=False \
    --preprocessing.path5=False\
    --preprocessing.filter.filter_size=3 \
    --preprocessing.bspline.spacing="[1, 1, 1]" \
    --preprocessing.bspline.size="[128, 128, 128]" \
    --write_index_order: "C"

python fetusnet.py \
    --mode=preprocess \
    --ds.sys="/media/yusuf/HDD 4TB/" \
    --ds.dataframe="sinMaternitat__fold0" \
    --preprocessing.save_dir="DATA_path5_unaligned" \
    --preprocessing.path1=False \
    --preprocessing.path2=False \
    --preprocessing.path3=False \
    --preprocessing.path4=False \
    --preprocessing.path5=True \
    --preprocessing.filter.filter_size=3 \
    --preprocessing.bspline.spacing="[1, 1, 1]" \
    --preprocessing.bspline.size="[128, 128, 128]" \
    --write_index_order: "F"