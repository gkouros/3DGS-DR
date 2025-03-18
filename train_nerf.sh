#!/bin/bash
set -e
python -u train.py -s data/nerf_synthetic/lego   -m logs/baseline/nerf_synthetic/lego   --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/drums  -m logs/baseline/nerf_synthetic/drums  --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/ship   -m logs/baseline/nerf_synthetic/ship   --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/hotdog -m logs/baseline/nerf_synthetic/hotdog --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/ficus  -m logs/baseline/nerf_synthetic/ficus  --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/mic    -m logs/baseline/nerf_synthetic/mic    --eval --iterations 61000 --white_background --densification_interval_when_prop 100
