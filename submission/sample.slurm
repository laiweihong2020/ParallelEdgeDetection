#!/bin/bash
#SBATCH --error=err-%j.err
#SBATCH --output=batch_n16-%j.out
#SBATCH --partition=snowy
#SBATCH --time=0:45:00
#SBATCH --nodes=16
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH --job-name=batch_n16

module load gcc
module load openmpi

echo "16 nodes"
mpicxx -o par.out par_canny.cpp -fopenmp
mpirun -n 16 par.out