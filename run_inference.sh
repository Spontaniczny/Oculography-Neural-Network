python -m inference.run_inference \
    --dataset_path "datasets/annotated_data/2022_06_01_A_video/frames" \
    --model_config "saved_models/u_net/03:01:2025-21:44:38.json" \
    --save_folder "3" \
    --save_step 1 \
    --remove_artifacts True \