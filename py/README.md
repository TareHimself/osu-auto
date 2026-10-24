# osu-auto (python)

Map picking, data export, training, evaluation and ONNX export. Run everything from `py/` with `uv run python -m osu_auto.<module>`. The pipeline and the commands are in the [top-level README](../README.md).

## Modules

**Maps**

| module | what it does |
|---|---|
| `maps` | picks ranked maps over a star rating, CS and AR grid, downloads them, writes a job list |
| `synth` | generates synthetic maps with flat position, jump and timing distributions |

**Export** (talks to the C# host in `host/`)

| module | what it does |
|---|---|
| `host` | ZeroMQ client for the host's `serve` command |
| `export_jobs` | runs a job list across parallel hosts and skins: labels, then frames |
| `export_frames` | frames for one label export, rendered and downsampled in the host |
| `export_status` | small window showing the progress of a running `export_jobs` |
| `labels` | reads a label export (cursor, keys, hit objects, meta) |
| `framestore` | zstd-compressed frame files |
| `cursor` | the skin's cursor sprite, measured from the host |
| `geometry` | playfield, window and capture box coordinates |

**Training**

| module | what it does |
|---|---|
| `train_joint` | trains the model; its defaults are the current best recipe |
| `dataset` | loads a frame export and its per-step targets (aim, press, down, spinner, strain) |
| `gpu_data` | GPU-resident batches: strain-weighted sampling, cursor compositing, latency and spinner recovery augmentation |
| `models` | `FrameNet` and `load_checkpoint` |
| `targets` | aim encoding between playfield pixels and model outputs |
| `tracking` | optional wandb logging and Hugging Face Hub storage |

**Evaluation and playing**

| module | what it does |
|---|---|
| `evaluate` | closed loop in the host: the model plays a map and the game judges it |
| `env` | step-by-step game env over the host |
| `keys` | turns press and hold probabilities into key presses, shared by every player |
| `export_onnx` | ONNX export for the runner, checked against PyTorch in pixels and click probability |
| `promote` | makes a run the default model (`data/models/current`) |
| `play` | Python player for the real game (the native runner in `play/` replaces it for normal use) |
| `capture` | Windows Graphics Capture of a game window |

**Checks** (debugging the host and capture, not part of the pipeline)

| module | what it checks |
|---|---|
| `check_host` | the host control channel and play area |
| `check_env` | the env with autoplay positions gives autoplay-quality judgements |
| `check_capture` | captures after seeks, with the computed geometry overlaid |
| `check_readback` | the host's in-process readback against a window capture |
| `plot_labels` | sanity plots of a label export |
| `make_runner_fixture`, `make_shader_fixture` | test data for `runner verify` and `runner shader-test` |

## Conventions

- Paths default to folders under the repo's `data/` (`paths.py`), so commands work from any clone.
- A checkpoint (`joint_best.pt`) holds the weights plus `stack`, `width`, `aim_head`, `frame_hw`, `latency_max_ms` and `offsets`. Load it with `models.load_checkpoint`.
- Model outputs are `(x, y, press logit, down logit)`. `x, y` use the aim encoding in `targets.py`.
