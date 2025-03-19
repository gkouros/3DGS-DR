#!/bin/bash
set -e
RESOLUTION=${1:-1}
GUI=${2:-}
python -u train.py -s data/ref_real/gardenspheres -m logs/baseline_res${RESOLUTION}/ref_real/gardenspheres --eval ${GUI} -r ${RESOLUTION} --iterations 61000  --longer_prop_iter 36_000 --use_env_scope --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974
python -u train.py -s data/ref_real/sedan         -m logs/baseline_res${RESOLUTION}/ref_real/sedan         --eval ${GUI} -r ${RESOLUTION} --iterations 61000 --longer_prop_iter 36_000 --use_env_scope --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138
python -u train.py -s data/ref_real/toycar        -m logs/baseline_res${RESOLUTION}/ref_real/toycar        --eval ${GUI} -r ${RESOLUTION} --iterations 61000  --longer_prop_iter 36_000 --use_env_scope --env_scope_center 0.6810 0.8080 4.4550 --env_scope_radius 2.707
