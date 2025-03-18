#!/bin/bash
set -e
RESOLUTION=${1:-1}
python -u eval.py -m logs/baseline_low_res/ref_real/gardenspheres -r ${RESOLUTION} --save_images
python -u eval.py -m logs/baseline_low_res/ref_real/sedan         -r ${RESOLUTION} --save_images
python -u eval.py -m logs/baseline_low_res/ref_real/toycar        -r ${RESOLUTION} --save_images