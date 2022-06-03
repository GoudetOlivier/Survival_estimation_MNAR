#!/bin/bash

#SBATCH --array=0-199
#SBATCH --time=9:59:00
#SBATCH -N1
#SBATCH --no-kill
#SBATCH --error=slurm-err-%j.out
#SBATCH --output=slurm-o-%j.out	
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000



python main_MNAR.py $1 $SLURM_ARRAY_TASK_ID $2 'MAR001' $3
