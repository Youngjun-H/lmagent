#!/bin/bash
#SBATCH --job-name=token_analysis_18gpu
#SBATCH --output=logs/token_analysis_18gpu_%j.out
#SBATCH --error=logs/token_analysis_18gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:18
#SBATCH --partition=hopper

# 로그 디렉토리 생성
mkdir -p logs

# 작업 시작 시간 기록
echo "=== 작업 시작: $(date) ==="
echo "작업 ID: $SLURM_JOB_ID"
echo "노드: $SLURM_NODELIST"
echo "할당된 노드 수: $SLURM_NNODES"
echo "총 작업 수: $SLURM_NTASKS"

# GPU 정보 확인
echo "=== GPU 할당 정보 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# 환경 설정
module purge
module load python/3.12
module load cuda/12.0

# 가상환경 활성화 (필요한 경우)
# source /path/to/your/venv/bin/activate

# srun으로 Python 스크립트 실행 (18개 작업 병렬 실행)
srun python token_analysis.py

# 작업 완료 시간 기록
echo "=== 작업 완료: $(date) ===" 