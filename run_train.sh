python -m train.init_training \
    --net_type segmentation \
    --backbone mobile_net_small \
    --input_size 128 \
    --dataset datasets/rat_eye/^.*$ \
    --loss_type  dice \
    --optimizer AdamW \
    --patience 10 \
    --max_epochs 100 \
    --batch_size 128