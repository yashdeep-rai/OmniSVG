#!/bin/bash
#SBATCH --job-name=omnisvg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=omnisvg_%j.out
#SBATCH --error=omnisvg_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting OmniSVG inference..."

# Activate environment
source ~/venvs/omnisvg/bin/activate

# Move to project directory
cd ~/My_Projects/OmniSVG

# Display GPU information
nvidia-smi

# Run inference
python inference.py \
    --task text-to-svg \
    --input test_prompt.txt \
    --output output_complex \
    --model-size 8B \
    --model-path ~/My_Projects/models/Qwen2.5-VL-7B-Instruct \
    --weight-path ~/My_Projects/models/OmniSVG1.1_8B/pytorch_model.bin \
    --num-candidates 1 \
    --max-length 512 \
    --temperature 0.3 \
    --top-p 0.9 \
    --top-k 30 \
    --verbose

echo "Inference completed."