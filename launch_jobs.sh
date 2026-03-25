#!/bin/bash
DATE=`date -I`
TIME=`date +'%H-%M-%S'`
for SEED in {1..5};
do
for CONFIG in\
    test_eig.yaml\
    test_mutual_information_ess.yaml\
    test_mutual_information_ep.yaml
do
mkdir -p ~/bayes-phase/multirun/$DATE/$TIME/$CONFIG
echo "#BSUB -cwd '/home/research/b.huggins/bayes-phase/'
#BSUB -q 'cpu-compute normal' 
#BSUB -outdir './multirun/$DATE/$TIME/$CONFIG'
#BSUB -o './multirun/$DATE/$TIME/$CONFIG/bsub_%J.log'
#BSUB -R '(!gpu)'
#BSUB -N 
#BSUB -J 'bayes_phase_$CONFIG_$SEED'
#BSUB -R 'rusage[mem=8GB,swp=4GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -n 4
#BSUB -g /bayes_phase_test
micromamba run -p /home/research/b.huggins/.local/share/mamba/envs/bayes-phase uv run python -u bayes_phase --config bayes_phase/active_learning/config/$CONFIG --overrides global_seed=$SEED" | bsub
done
done
