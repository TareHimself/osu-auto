# Job lists

A job list is a JSON array of exports to make: `{map, diff, mods?, spin_recovery?}`, with `map` a file under `data/beatmaps/`.

`v3_all.json` is the training set: 86 jobs on 77 ranked maps and 96 on 78 synthetic maps. From `py/`:

- `uv run python -m osu_auto.maps --fetch ../data/jobs/v3_all.json` downloads the ranked maps (named by beatmapset id).
- `uv run python -m osu_auto.synth --config ../configs/v4.yaml` generates the synthetic maps. It is seeded, so the files are identical on every machine.

Other lists you make (`maps --jobs`, `synth --jobs`) and the export lists for training (`export_jobs --list`) stay local.
