python -m inference.run_inference \
    --dataset_path "datasets/annotated_data/2022_09_02_A_video/frames" \
    --model_config "saved_models/deeplab/02:01:2025-19:56:36.json" \
    --save_folder "3" \
    --save_step 1000 \
    --remove_artifacts True \