#!/bin/bash
#SBATCH --job-name=token_analysis
#SBATCH --output=logs/token_analysis_%j.out
#SBATCH --error=logs/token_analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper

# 로그 디렉토리 생성
mkdir -p logs

# 작업 시작 시간 기록
echo "=== 작업 시작: $(date) ==="
echo "작업 ID: $SLURM_JOB_ID"
echo "노드: $SLURM_NODELIST"

# srun으로 Python 스크립트 실행
srun python save_vocab_korean.py

# 작업 완료 시간 기록
echo "=== 작업 완료: $(date) ===" 