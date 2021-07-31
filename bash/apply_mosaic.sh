

SOURCE_DIR='source'
TARGET_DIR='target'
ALPHA=0.7

python src/apply_mosaic.py \
    --source_dir $SOURCE_DIR \
    --target_dir $TARGET_DIR \
    --alpha $ALPHA \
    --grid_size 25 25 \
    --num_clusters 5 \
    --output_dir output