CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain BTCV  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run1' 
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain BTCV  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run2'
# CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain BTCV  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run3'

CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain CHAOS  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run1'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain CHAOS  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run2'
# CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain CHAOS  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run3'

CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain IRCAD  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run1'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain IRCAD  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run2'
# CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain IRCAD  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run3'

CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain LITS  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run1'
CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain LITS  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run2'
# CUDA_VISIBLE_DEVICES=2 python main_seg_mame.py --target_domain LITS  --method MET --a0 0 --a1 1 --a2 0.05 --suffix 'warmup_mask2_run3' 
