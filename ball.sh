python -u train.py -s data/ref_shiny/ball    -m logs/baseline/ref_shiny/ball    --eval --iterations 61000 --white_background
python -u eval.py -m logs/baseline/ref_shiny/ball    --save_images
