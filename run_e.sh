#!/bin/bash
#SBATCH --job-name=bert-e
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

P=/projects/fakenewsnlp
export TMPDIR=$P/.tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR
trap 'rm -rf $TMPDIR' EXIT
export HF_HOME=$P/.tmp/hf HF_HUB_OFFLINE=1

PY=$P/envs/nlp/bin/python
cd $P/NLP-Project
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
$PY -m scripts.run_transformers --author "Menujaa" "$@"