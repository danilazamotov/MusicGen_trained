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
