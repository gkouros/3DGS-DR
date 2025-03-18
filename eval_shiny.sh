#!/bin/bash
set -e
python -u eval.py -m logs/baseline/ref_shiny/ball    --save_images
python -u eval.py -m logs/baseline/ref_shiny/car     --save_images
python -u eval.py -m logs/baseline/ref_shiny/coffee  --save_images
python -u eval.py -m logs/baseline/ref_shiny/helmet  --save_images
python -u eval.py -m logs/baseline/ref_shiny/teapot  --save_images
python -u eval.py -m logs/baseline/ref_shiny/toaster --save_images