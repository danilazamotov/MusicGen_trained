# MusicGen_trained
This is my instruction version of MusicGen  training (audiocraft).

Installing audiocraft

- model/lm/model_scale=small (300M) ~ 14-24G gpu (it depends on the batch size (1-6) in config)
- medium (1.5B) > 38g gpu
- large (3.3B) over 40g+ gpu

Google colab example in terminal

```jsx
cd /content/drive/MyDrive/audiocraft_training
git clone https://github.com/GrandaddyShmax/audiocraft_plus.git
cd audiocraft_plus
apt install python3.10-venv
python -m venv venv
source venv/bin/activate
pip install -e .
python3 -m pip install setuptools wheel
sudo apt-get update
sudo apt-get install sox libsox-dev libsox-fmt-all
sudo apt-get install libopenblas-base libopenblas-dev
sudo apt-get install ffmpeg
pip install transformers --upgrade
pip install  torchmetrics --upgrade
pip install -U torchaudio
pip uninstall xformers
pip install xformers
```

# Dataset
- Use .ipynb dataset processing to split audio into chunks
# Creating manifest files

```jsx
python -m audiocraft.data.audio_dataset dataset/example egs/example/data.jsonl
```
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
  - override /dset: audio/default
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
  batch_size: 4 #64  # 32 GPUs
  sample_on_weight: false  # Uniform sampling all the way
  num_workers: 10
  shuffle: true
  segment_duration: 15
  min_segment_ratio: 0.8
  sample_on_duration: false  # Uniform sampling all the way
    
generate:
  lm:
    use_sampling: true
    top_k: 250
    top_p: 0.0

optim:
  epochs: 200
  updates_per_epoch: 1000
  lr: 1e-4
  optimizer: adamw
  adam:
    betas: [0.9, 0.95]
    weight_decay: 0.01
    eps: 1e-8

logging:
  log_tensorboard: true

schedule:
  lr_scheduler: cosine
  cosine:
    warmup: 1000
    lr_min_ratio: 0.0
    cycle_length: 1.0

checkpoint:
  save_last: true
  save_every: 10
  keep_last: 2
  keep_every_states: null
```

# bash  dora start!
```jsx
echo "Starting TensorBoard..." >> run.log
tensorboard --logdir=/mnt/checkpoint_audiocraft_trained/audiocraft_user/xps/ --port=50500 --bind_all >> run.log 2>&1 &
echo "TensorBoard is running at [http://localhost:50500](http://localhost:50500/)" >> run.log

echo "Running Dora..." >> run.log
export USER=user
export XFORMERS_MORE_DETAILS=1
export HYDRA_FULL_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export OC_CAUSE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
dora --verbose run dset=audio/example \
solver=musicgen/musicgen_base_32khz \
model/lm/model_scale=small \
continue_from=//pretrained/facebook/musicgen-small
conditioner=text2music \
evaluate.metrics.chroma_cosine=true >> run.log 2>&1
echo "Dora process finished." >> run.log

echo "Deactivating environment..." >> run.log
deactivate >> run.log
echo "Environment deactivated." >> run.log

kill $!  # Убиваем последний фоновый процесс (TensorBoard)
echo "TensorBoard has been stopped." >> run.log
```

# Correct files 

I've had some bugs, my opinion is what needs to be changed
/audiocraft/data/audio_utils.py

```jsx
def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to int 16 bits PCM format.

    ..Warning:: There exist many formula for doing this conversion. None are perfect
    due to the asymmetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistencies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    """
    if wav.dtype.is_floating_point:
        max_val = wav.abs().max()
        if max_val > 1:
            wav = wav / max_val  # Normalize to range [-1, 1]
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:  # clipping would occur
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav
```
/audiocraft/app.py
- 914
```jsx
 if MODEL.max_duration is None:
        print("MODEL.max_duration is not initialized or is None")
    if overlap is None:
        print("overlap is not initialized or is None")
        
    extend_stride = MODEL.max_duration - overlap if MODEL.max_duration and overlap else 0

    outs, outs_audio, outs_backup, input_length = _do_predictions(
        gen_type, [texts], [melody], sample, trim_start, trim_end, duration, image, height, width, background, bar1, bar2, channel, sr_select, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef, extend_stride=extend_stride)
    tags = [str(global_prompt), str(bpm), str(key), str(scale), str(raw_texts), str(duration), str(overlap), str(seed), str(audio_mode), str(input_length), str(channel), str(sr_select), str(model_shrt), str(custom_model_shrt), str(decoder), str(topk), str(topp), str(temperature), str(cfg_coef), str(gen_type)]
    wav_target, mp4_target, json_target = save_outputs(outs[0], outs_audio[0], tags, gen_type);
```

audiocraft_plus\audiocraft\models\musicgen.py
```jsx
def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        if max_duration is None:
            if hasattr(lm, 'cfg'):
                max_duration = lm.cfg.dataset.segment_duration  # type: ignore
            else:
                raise ValueError("You must provide max_duration when building directly MusicGen")

        self.max_duration: float = 30
```

```jsx
#assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
```
