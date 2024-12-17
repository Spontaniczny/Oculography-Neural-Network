python -m train.fine_tune \
    --net_config_file saved_models/deeplab/17:12:2024-00:40:58.json \
    --input_size 256 \
    --dataset "datasets/rat_eye/01|02" \
    --loss_type dice \
    --optimizer AdamW \
    --patience 30 \
    --max_epochs 200 \
    --batch_size 32