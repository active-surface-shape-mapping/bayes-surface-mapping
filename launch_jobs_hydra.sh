#!/bin/bash
DATE=`date -I`
TIME=`date +'%H-%M-%S'`
for EXPERIMENT in\
    eig\
    mi_ess\
    mi_ep
do
BASEPATH=/engrfs/project/garnett/bayes_phase/results/$DATE/$TIME/$EXPERIMENT
mkdir -p $BASEPATH
echo "#BSUB -cwd '/engrfs/project/garnett/bayes_phase/bayes-phase/'
#BSUB -q 'cpu-compute'
#BSUB -outdir '$BASEPATH'
#BSUB -o '$BASEPATH/bsub_%J.log'
#BSUB -R '(!gpu)'
#BSUB -N
#BSUB -J 'bayes_phase_$EXPERIMENT'
#BSUB -R 'rusage[mem=64,swp=4GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -n 20
#BSUB -g /bayes_phase
micromamba run -p /engrfs/project/garnett/mamba/envs/bayes-phase uv run python -u -m bayes_phase -m experiment=$EXPERIMENT experiment.global_seed='range(0,10)' '++hydra.sweep.dir=$BASEPATH'" | bsub
sleep 3
done
