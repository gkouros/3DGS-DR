#!/bin/bash
set -e
python -u train.py -s data/glossy_synthetic/angel  -m logs/baseline/glossy_synthetic/angel   --eval --iterations 61000 --white_background --longer_prop_iter 36_000
python -u train.py -s data/glossy_synthetic/bell   -m logs/baseline/glossy_synthetic/bell    --eval --iterations 91000 --white_background --longer_prop_iter 48_000  --opac_lr0_interval 0
python -u train.py -s data/glossy_synthetic/cat    -m logs/baseline/glossy_synthetic/cat     --eval --iterations 61000 --white_background
python -u train.py -s data/glossy_synthetic/horse  -m logs/baseline/glossy_synthetic/horse   --eval --iterations 61000 --white_background --longer_prop_iter 36_000
python -u train.py -s data/glossy_synthetic/luyu   -m logs/baseline/glossy_synthetic/luyu    --eval --iterations 61000 --white_background
python -u train.py -s data/glossy_synthetic/potion -m logs/baseline/glossy_synthetic/potion  --eval --iterations 61000 --white_background --longer_prop_iter 24_000
python -u train.py -s data/glossy_synthetic/tbell  -m logs/baseline/glossy_synthetic/tbell   --eval --iterations 61000 --white_background  --longer_prop_iter 36_000  --opac_lr0_interval 0
python -u train.py -s data/glossy_synthetic/teapot -m logs/baseline/glossy_synthetic/teapot  --eval --iterations 61000 --white_background --longer_prop_iter 36_000