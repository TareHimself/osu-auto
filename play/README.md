# Play

Plays osu!lazer with a trained model at 100 Hz, with no Python in the loop. It captures the game window, runs the ONNX model, smooths the aim, moves the cursor and presses Z/X. It does the same job as `py/src/osu_auto/play.py`, with steadier timing.

Windows only, x64, an NVIDIA GPU for CUDA (or `--cpu`), lazer in windowed or borderless mode.

## One time setup

```
uv run --project play/shell silo profile detect     # once per machine, finds MSVC
```

Pick the model to run (from `py/`). Training already writes the ONNX export next to the checkpoint; if a run lacks it, run `uv run python -m osu_auto.export_onnx ../data/models/<run>/joint_best.pt` first.

```
cd py
uv run python -m osu_auto.promote joint_v3_skins3
```

This copies the run's `joint_best.onnx`, weights and json into `data/models/current/`, which is what the runner loads by default.

## Play

From the repo root:

```
uv run --project play/shell silo run -C play/shell runner play
```

With no model path it loads `data/models/current/joint_best.onnx` and prints which run that is. Put a trained run there with `python -m osu_auto.promote <run>` (from `py/`). To run something else, pass the path: `runner play data/models/<run>/joint_best.onnx`.

The first run builds the runner and downloads onnxruntime, cuDNN and cuBLAS (about 2 GB, cached by silo). Later runs start straight away.

1. Start the map in lazer and click into the lazer window.
2. Press **F8** to start the agent. Press **F8** again to stop it.
3. Press **F9** to quit the runner.

The cursor and keys are only driven while the lazer window is focused. If it loses focus, the runner releases any held key, prints "game window not focused, input paused" and sends nothing until focus returns.

The first CUDA start spends a few seconds on cuDNN setup before the agent can run. Do not run training or the host while you play, they cause stalls.

### Options for `play`

| option | meaning |
|---|---|
| `--start-now` | start immediately instead of waiting for F8 |
| `--duration S` | stop and quit after S seconds of the agent running |
| `--no-inject` | run everything but send no input (to look at the numbers) |
| `--cpu` | run the model on the CPU |
| `--capture hook` | read frames from the capture hook instead of Windows capture (see below) |
| `--window P` | window whose title starts with P (default `osu!`) |
| `--keys zx` | the two keyboard keys |
| `--latency-extra MS` | extra delay told to the model on top of the measured one (default 8) |
| `--smooth-cutoff X`, `--smooth-beta X`, `--smooth-dcut X` | 1 euro filter, 0 cutoff turns smoothing off |
| `--log-dir D` | where logs go (default `data/play_logs` under the current folder, so run from the repo root) |

While running it prints one status line per second: capture fps, frame age, loop time, ticks, presses and the worst stalls.

## Logs

Each session writes two files to `data/play_logs`, in the same format as `play.py`:

- `play-<stamp>.jsonl`: `init`, `start`, `stop` and one `stats` line per second.
- `play-<stamp>-trace.npz`: per tick `t, x, y, p_press, p_down, k1, k2, frame_age_ms, infer_ms`.

Tick gap p99 should stay near 10 ms. Anything much larger means something else on the machine is stealing time.

## Capture hook (lower latency)

By default frames come from Windows capture. The hook copies each frame straight out of lazer's own `Present` call instead, which roughly halves frame age (about 2 ms against 4 ms). It loads a small DLL into the lazer process, like OBS game capture, and only reads pixels.

```
uv run --project play/shell silo run -C play/shell runner hook inject
uv run --project play/shell silo run -C play/shell runner play <model.onnx> --capture hook
uv run --project play/shell silo run -C play/shell runner hook eject
```

- Inject before starting the runner, eject when you are done. Closing lazer also removes it.
- Lazer has to be on the Direct3D 11 renderer, which is what `Renderer = Automatic` picks on Windows. Check with `runner hook-capture "osu!" 5`.
- Antivirus may flag the injection.
- Its own log is `%TEMP%\osu-auto-hook.log`.

## Checks

| command | what it checks |
|---|---|
| `runner verify <model.onnx> <fixture dir> [cpu\|cuda]` | model output, smoothing and key logic against Python on a recorded fixture |
| `runner shader-test <dir>` | the GPU gray and downsample step against cv2 on a saved frame |
| `runner capture "osu!" 5 out.pgm` | Windows capture for a few seconds, saves the last frame |
| `runner hook-capture "osu!" 5 out.pgm` | the same through the hook |

Fixtures come from Python, run from `py/`:

```
python -m osu_auto.make_runner_fixture <frame export> <out dir> --model <checkpoint.pt>
python -m osu_auto.make_shader_fixture <out dir>      # needs lazer open
python -m osu_auto.play --dump-frames <dir> ...       # record a live session instead
```

## Developing

The code, recipes and build notes are in `runner/`, see its README. `shell/` is only the project you run it from.

## Troubleshooting

- **No window starting with the given title**: lazer is closed, minimised, or the title does not start with `osu!`. Use `--window`.
- **Keys do nothing**: fullscreen breaks key injection, use windowed or borderless.
- **`the capture hook is not loaded`**: run `hook inject` first.
- **Frame age or tick gap jumps**: close other GPU or CPU heavy programs.
