#!/bin/bash -l

DIR=$( realpath -e -- $( dirname -- ${BASH_SOURCE[0]}))
cd $DIR
echo $DIR

sbatch large_shiny.job
sbatch large_glossy.job
