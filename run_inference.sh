python -m inference.run_inference \
    --dataset_path "datasets/extracted_frames/2022_06_01_A_video" \
    --model_config "saved_models/deeplab/21:12:2024-22:04:45.json" \
    --save_folder "2" \
    --save_step 100 \
    # --remove_artifacts True \