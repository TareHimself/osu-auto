"""Optional experiment tracking (wandb) and Hugging Face Hub storage. Everything is a no-op unless asked for."""
from __future__ import annotations

import json
from pathlib import Path


class Tracker:
    def __init__(self, project: str | None, name: str | None, config: dict, out_dir: Path):
        self.run = None
        self.out_dir = out_dir
        if project:
            import wandb

            self.run = wandb.init(project=project, name=name, config=config, dir=str(out_dir), job_type="train", tags=["train"])

    def log(self, stats: dict, step: int | None = None) -> None:
        if self.run:
            self.run.log(stats, step=step)

    def artifact(self, path: Path, name: str, kind: str = "model", metadata: dict | None = None) -> None:
        if self.run:
            import wandb

            art = wandb.Artifact(name, type=kind, metadata=metadata or {})
            art.add_file(str(path))
            self.run.log_artifact(art)

    def finish(self, summary: dict | None = None) -> None:
        if self.run:
            for k, v in (summary or {}).items():
                self.run.summary[k] = v
            self.run.finish()

    @property
    def id(self) -> str | None:
        return self.run.id if self.run else None


def log_eval(model_dir: Path, result: dict, wandb_project: str | None = None) -> None:
    """Append a closed-loop eval result to evals.json next to the checkpoint, and to the training run on wandb if it has one."""
    path = model_dir / "evals.json"
    evals = json.loads(path.read_text()) if path.exists() else []
    evals.append(result)
    path.write_text(json.dumps(evals, indent=2))
    run_file = model_dir / "wandb_run.json"
    if wandb_project and run_file.exists():
        import wandb

        info = json.loads(run_file.read_text())
        run = wandb.init(project=wandb_project, id=info["id"], resume="allow")
        run.log({f"eval/{result['map']}/{k}": v for k, v in result.items() if isinstance(v, (int, float))})
        run.finish()


def push_folder(folder: Path, repo_id: str, repo_type: str = "dataset", path_in_repo: str | None = None) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=True)
    api.upload_large_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder)) if path_in_repo is None else \
        api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder), path_in_repo=path_in_repo)
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"


def pull_folder(repo_id: str, local: Path, repo_type: str = "dataset", allow: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(local), allow_patterns=allow))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Push or pull export sets and models to the Hugging Face Hub")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("push")
    a.add_argument("folder")
    a.add_argument("repo_id", help="e.g. TareHimself/osu-auto-exports")
    a.add_argument("--type", choices=["dataset", "model"], default="dataset")
    a.add_argument("--path-in-repo", default=None, help="subfolder in the repo; default uploads the folder as the repo root")
    b = sub.add_parser("pull")
    b.add_argument("repo_id")
    b.add_argument("local")
    b.add_argument("--type", choices=["dataset", "model"], default="dataset")
    b.add_argument("--allow", default=None, help="glob of files to fetch, e.g. 'angelmegumin-ai-friendly/*'")
    args = p.parse_args()
    if args.cmd == "push":
        print(push_folder(Path(args.folder), args.repo_id, args.type, args.path_in_repo))
    else:
        print(pull_folder(args.repo_id, Path(args.local), args.type, args.allow))


if __name__ == "__main__":
    main()
