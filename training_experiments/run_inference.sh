python -m inference.run_inference \
    --dataset_path "datasets/extracted_frames/2022_06_01_A_video" \
    --model_config "saved_models/deeplab/12:01:2025-00:29:37.json" \
    --save_folder "10" \
    --save_step 100 \
    --remove_artifacts True \