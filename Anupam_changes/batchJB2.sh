#! /bin/bash -l

#$ -pe omp 4
#$ -P cs598
#$ -l h_rt=72:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -l gpu_type=A100|L40S
#$ -N MML_llava_finetuning3
#$ -j y
#$ -m ea
#$ -o outputs/MML_llava_finetuning3.out

module load python3/3.10.12
source /projectnb/cs598/students/achetia/venvs/AJ/bin/activate
python MML_2.py exp_name=MML_llava_finetuning3
