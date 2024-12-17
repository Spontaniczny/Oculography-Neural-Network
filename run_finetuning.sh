python -m train.fine_tune \
    --net_config_file saved_models/ellipsenet/17:12:2024-14:32:36.json \
    --input_size 256 \
    --dataset "datasets/rat_eye/01|02" \
    --loss_type smooth_l1 \
    --optimizer AdamW \
    --patience 30 \
    --max_epochs 200 \
    --batch_size 32