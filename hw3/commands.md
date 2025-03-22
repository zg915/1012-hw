ssh burst

srun --account=ds_ga_1012-2025sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash