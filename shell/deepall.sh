# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain VGH  --method deepall  --suffix 'run1' 
# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain VGH  --method deepall  --suffix 'run2'
CUDA_VISIBLE_DEVICES=0 python main_mame.py --target_domain VGH  --method deepall  --suffix 'run3'

# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain NKI  --method deepall  --suffix 'run1'
# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain NKI  --method deepall  --suffix 'run2'
CUDA_VISIBLE_DEVICES=0 python main_mame.py --target_domain NKI  --method deepall  --suffix 'run3'

# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain IHC  --method deepall  --suffix 'run1'
# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain IHC  --method deepall  --suffix 'run2'
# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain IHC  --method deepall  --suffix 'run3'

# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain NCH  --method deepall  --suffix 'run1'
# CUDA_VISIBLE_DEVICES=2 python main_mame.py --target_domain NCH  --method deepall  --suffix 'run2'
CUDA_VISIBLE_DEVICES=0 python main_mame.py --target_domain NCH  --method deepall  --suffix 'run3'
