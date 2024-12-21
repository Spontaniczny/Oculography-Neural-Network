python -m train.fine_tune \
    --net_config_file saved_models/ellipsenet/21:12:2024-00:40:48.json \
    --input_size 256 \
    --dataset "datasets/annotated_data/2022_06_01_A_video" \
    --loss_type smooth_l1_area \
    --optimizer AdamW \
    --patience 10 \
    --max_epochs 100 \
    --batch_size 16