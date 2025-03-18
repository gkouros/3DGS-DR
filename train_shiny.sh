#!/bin/bash
set -e
python -u train.py -s data/ref_shiny/ball    -m logs/baseline/ref_shiny/ball    --eval --iterations 61000 --white_background
python -u train.py -s data/ref_shiny/car     -m logs/baseline/ref_shiny/car     --eval --iterations 61000 --white_background
python -u train.py -s data/ref_shiny/coffee  -m logs/baseline/ref_shiny/coffee  --eval --iterations 61000 --white_background
python -u train.py -s data/ref_shiny/helmet  -m logs/baseline/ref_shiny/helmet  --eval --iterations 61000 --white_background
python -u train.py -s data/ref_shiny/teapot  -m logs/baseline/ref_shiny/teapot  --eval --iterations 61000 --white_background
python -u train.py -s data/ref_shiny/toaster -m logs/baseline/ref_shiny/toaster --eval --iterations 61000 --white_background --longer_prop_iter 24_000