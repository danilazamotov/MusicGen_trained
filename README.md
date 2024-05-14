# MusicGen_trained
This is my instruction version of MusicGen  training (audiocraft).

Installing audiocraft

- model/lm/model_scale=small (300M) ~ 14-24G gpu (it depends on the batch size (1-6) in config)
- medium (1.5B) > 38g gpu
- large (3.3B) over 40g+ gpu

Google colab example in terminal

- cd /content/drive/MyDrive/audiocraft_training
- git clone https://github.com/GrandaddyShmax/audiocraft_plus.git
- cd audiocraft_plus
- apt install python3.10-venv
- python -m venv venv
- source venv/bin/activate
- pip install -e .
- python3 -m pip install setuptools wheel
- sudo apt-get update
- sudo apt-get install sox libsox-dev libsox-fmt-all
- sudo apt-get install libopenblas-base libopenblas-dev
- sudo apt-get install ffmpeg
- pip install transformers --upgrade
- pip install  torchmetrics --upgrade
- pip install -U torchaudio
- pip uninstall xformers
- pip install xformers

# Dataset
- Use .ipynb dataset processing to split audio into chunks
# Creating manifest files
- python -m audiocraft.data.audio_dataset dataset/example egs/example/data.jsonl

# Config

/config/teams
```jsx
default:
  dora_dir: /mnt/audiocraft_${oc.env:USER} / полный путь, где будут сохраняться чекпоинты
  partitions:
    global: debug # Уровень дебагинга, / info
    team: debug
  reference_dir: /mnt
darwin:  # if we detect we are on a Mac, then most likely we are doing unit testing etc.
  dora_dir: /tmp/audiocraft_${oc.env:USER}
  partitions:
    global: debug
    team: debug
  reference_dir: /tmp
```

# The optimal config, in my opinion, for the MusicGen model (text-to-music)
```jsx
# @package __global__

# This is the training loop solver
# for the base MusicGen model (text-to-music)
# on monophonic audio sampled at 32 kHz
defaults:
  - musicgen/default
  - /model: lm/musicgen_lm
  # - override /dset: audio/default
  - override /dset: audio/example
  #- override /model/lm/model_scale: medium
  - _self_

autocast: true
autocast_dtype: float16

# EnCodec large trained on mono-channel music audio sampled at 32khz
# with a total stride of 640 leading to 50 frames/s.
# rvq.n_q=4, rvq.bins=2048, no quantization dropout
# (transformer_lm card and n_q must be compatible)
compression_model_checkpoint: //pretrained/facebook/encodec_32khz

channels: 1
sample_rate: 32000

deadlock:
  use: true  # deadlock detection

dataset:
  batch_size: 8  # 32 GPUs  Для RTX 4090, но лучше 1-4
  sample_on_weight: false  # Uniform sampling all the way
  sample_on_duration: false  # Uniform sampling all the way
  num_workers: 20 # Вроде как зависит от процессора
  
generate:
  every: 25
  num_workers: 20
  path: samples
  audio:
    format: wav
    strategy: loudness
    sample_rate: ${sample_rate}
    loudness_headroom_db: 14
  lm:
    prompted_samples: true
    unprompted_samples: true
    gen_gt_samples: false
    prompt_duration: null
    gen_duration: null
    remove_prompts: false
    use_sampling: true
    temp: 1.0
    top_k: 250
    top_p: 0.1

# optim:
#   epochs: 100
#   oprimazer: adamw
#   max_norm: 0.5
#   lr: 1
#   ema:
#     use: true
#     updates: 10
#     device: cuda

optim:
  epochs: 200
  updates_per_epoch: 2000
  lr: 1e-4
  optimizer: adamw
  max_norm: 1.0
  eager_sync: true
  adam:
    betas: [0.9, 0.95]
    weight_decay: 0.1
    eps: 1e-8
  ema:
    use: true
    updates: 10
    device: cuda

logging:
  log_tensorboard: true

schedule:
  lr_scheduler: cosine
  cosine:
    warmup: 3000
    lr_min_ratio: 0.0
    cycle_length: 1.0

checkpoint:
  save_last: true
  save_every: 20
  keep_last: 10
  keep_every_states: null
```
