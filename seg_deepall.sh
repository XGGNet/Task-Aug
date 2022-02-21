CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain BTCV  --method deepall --suffix 'warmup_mask_run1' 
CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain BTCV  --method deepall --suffix 'warmup_mask_run2'
CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain BTCV  --method deepall --suffix 'warmup_mask_run3'

CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain CHAOS  --method deepall --suffix 'warmup_mask_run1'
CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain CHAOS  --method deepall --suffix 'warmup_mask_run2'
CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain CHAOS  --method deepall --suffix 'warmup_mask_run3'

CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain IRCAD  --method deepall --suffix 'warmup_mask_run1'
CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain IRCAD  --method deepall --suffix 'warmup_mask_run2'
CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain IRCAD  --method deepall --suffix 'warmup_mask_run3'

CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain LITS  --method deepall --suffix 'warmup_mask_run1'
CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain LITS  --method deepall --suffix 'warmup_mask_run2'
CUDA_VISIBLE_DEVICES=0 python main_seg_mame.py --target_domain LITS  --method deepall --suffix 'warmup_mask_run3'
