python -m train.fine_tune \
    --net_config_file saved_models/deeplab/18:12:2024-02:17:34.json \
    --input_size 256 \
    --dataset "datasets/annotated_data/2022_06_01_A_video" \
    --loss_type dice \
    --optimizer AdamW \
    --patience 30 \
    --max_epochs 200 \
    --batch_size 32