#!/bin/bash

GPU_ID=3  # Change this to the index of the GPU you want to monitor

while true; do
    gpu_utilization=$(nvidia-smi --id=$GPU_ID --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    if [ $gpu_utilization -eq 0 ]; then
        echo "GPU $GPU_ID is free. Starting training..."
        CUDA_VISIBLE_DEVICES=3 python main.py --c ./configs/config.yaml
        break  # Exit the loop once training is started
    fi
    echo "Waiting for available GPU..."
    sleep 300  # Sleep for 5 minutes (300 seconds) before checking again
done

