CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain BTCV  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run1' 
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain BTCV  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run2'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain BTCV  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run3'

CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain CHAOS  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run1'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain CHAOS  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run2'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain CHAOS  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run3'

CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain IRCAD  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run1'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain IRCAD  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run2'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain IRCAD  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run3'

CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain LITS  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run1'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain LITS  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run2'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain LITS  --method ET --a0 1 --a1 0 --a2 0 --suffix 'warmup_run3' 
