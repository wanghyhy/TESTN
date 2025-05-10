# Chicago
python train.py --dataset chicago --evaluate-every 10 --regularization 1e-4 --split_ratio 0.5 --ssl_weight 2.0

# Tokyo
python train.py --dataset tokyo --evaluate-every 10 --regularization 1e-5 --split_ratio 0.5 --ssl_weight 0.5

# New York
python train.py --dataset nyc --evaluate-every 10 --regularization 1e-3 --split_ratio 0.5 --ssl_weight 2.5