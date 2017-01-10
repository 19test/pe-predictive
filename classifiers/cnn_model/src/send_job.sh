#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name
#SBATCH --job-name=IMB4
#################
#a file for job output, you can check job progress
#SBATCH --output=IMB4.out
#################
# a file for errors from the job
#SBATCH --error=IMB4.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=20:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=2
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=4000
#you could use --mem-per-cpu; they mean what we are calling cores
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-task=16" instead
#SBATCH --ntasks-per-node=16
#################

srun -N 1 -p ibiis --gres=gpu:1 --time=24:00:00 --pty job.sh 
#srun -N 1 -p gpu --qos=gpu --gres=gpu:1 --time=24:00:00 --pty job.sh
