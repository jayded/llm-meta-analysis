#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=gpt35
#SBATCH --mem=45G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --output=exec.gpt35.%j.evaluate.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yun.hy@northeastern.edu

module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# outcome_types
python3 ../run_task.py --model gpt35 --task outcome_type --split test --prompt without-abstract-results --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/outcome_type

# binary outcomes
# python3 ../run_task.py --model gpt35 --task binary_outcomes --split test --prompt yaml --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/binary_outcomes

# continuous_outcomes
# python3 ../run_task.py --model gpt35 --task continuous_outcomes --split test --prompt yaml --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/continuous_outcomes