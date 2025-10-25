# 使用 HuggingFace 镜像站，解决连接不稳定问题
export HF_ENDPOINT=https://hf-mirror.com
# 增加超时时间
export HF_HUB_DOWNLOAD_TIMEOUT=300

# NCCL 调试和优化选项  
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
# 禁用 NVLS (NVLink Sharp) - H100 特性但在某些系统上可能不稳定
export NCCL_NVLS_ENABLE=0
# 使用标准 NVLink P2P 通信
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0

bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/delta_net-340M-4K-10B/batch1.seqlen65536.context4096.warmup1024.update1.steps20480.lr1e-3.cosine \
  --model.config configs/delta_net_340M.json \
  --model.tokenizer_path fla-hub/delta_net-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree 8 \
  --training.data_parallel_shard_degree 1 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2 \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1
