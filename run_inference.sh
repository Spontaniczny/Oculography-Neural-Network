python -m inference.run_inference \
    --dataset_path "datasets/extracted_frames/2022_06_01_A_video" \
    --model_config "saved_models/deeplab/20:12:2024-22:54:58.json" \
    --save_folder "13" \
    --save_images True \
    --max_images 200 \
    --remove_artifacts True \