"""
Experiment tracking and checkpoint provenance.

Two independent layers, deliberately decoupled:

A. **JSON manifest** - written next to every .pth checkpoint. Zero dependencies,
   always on. Answers "which hyperparameters, which code, which metrics produced
   this weight file?" Previously that information lived only in whatever value
   train.py happened to have at the time, so a checkpoint was untraceable the
   moment the config was edited.

B. **MLflow run** - optional. Lets you compare runs side by side in `mlflow ui`.
   MLflow is imported lazily and every call is wrapped, so training runs
   perfectly normally when mlflow is not installed or fails to start.

Nothing here is allowed to interrupt training: a tracking failure prints a
warning and is then ignored.
"""

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


# Bumped whenever the MEANING of the recorded losses changes, so numbers from
# different eras are never silently compared.
#   1 = classification loss summed over queries then divided by the matched
#       target count, no background down-weighting; topk_score_head received no
#       gradient and was therefore never trained.
#   2 = DETR-style weighted mean over all queries with eos_coef background
#       damping; encoder objectness loss added to supervise Mixed Query
#       Selection; background confirmed as class index 0 across train and
#       inference.
#   3 = bbox_head now predicts a centre OFFSET from the selected patch's centre
#       (reference point) rather than an absolute coordinate, so its weights are
#       not interchangeable with version 2 even though the tensor shapes match.
#   4 = three changes that each move the recorded numbers:
#       (a) the encoder objectness target went from "every patch whose centre is
#           inside a GT box" to a small block on each GT's centre, so `enc` is a
#           different quantity - and its pos_weight ceiling rose 100 -> 400;
#       (b) Hungarian matching now scores with the CONFIGURED loss weights and
#           with CIoU instead of plain IoU, so the assignment - and therefore
#           every box term computed from it - differs;
#       (c) ciou_loss_weight 2.0 -> 5.0 and aux_loss_weight 0 -> 0.4, which
#           together scale total loss by about 1.5x on identical predictions.
#       A version-3 best_val_loss is NOT beatable by a version-4 run; see the
#       resume guard in train.py.
#   5 = the inputs changed, not the loss formula, but every recorded number
#       moves with them:
#       (a) num_queries 200 -> 100, which halves the background mass in the
#           query-mean classification loss;
#       (b) images are resized onto one of three aspect-ratio canvases instead
#           of a square, so the encoder objectness grid is 57x43 / 50x50 / 43x57
#           and `enc` is computed over a different number of patches;
#       (c) training adds scale jitter + random crop, so train loss is measured
#           on a harder distribution than before (val is unaugmented, but its
#           canvas changed under (b)).
#       bbox_head still predicts an offset from the reference point, so version
#       4 weights load and train on without conversion - only the bookkeeping
#       thresholds reset.
LOSS_SEMANTICS_VERSION = 5

# Structure version of the manifest file itself.
MANIFEST_SCHEMA_VERSION = 1

DEFAULT_EXPERIMENT_NAME = 'dinov3-detr'


def _git_commit(repo_dir):
    """Return the current commit (with a -dirty suffix), or None outside git."""
    def _run(args):
        try:
            r = subprocess.run(args, cwd=repo_dir, capture_output=True,
                               text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    commit = _run(['git', 'rev-parse', 'HEAD'])
    if not commit:
        return None
    if _run(['git', 'status', '--porcelain']):
        commit += '-dirty'
    return commit


def collect_environment(repo_dir=None):
    """Snapshot of everything needed to reproduce a run."""
    repo_dir = repo_dir or os.getcwd()
    env = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'git_commit': _git_commit(repo_dir),
    }
    try:
        import torchvision
        env['torchvision'] = torchvision.__version__
    except Exception:
        env['torchvision'] = None
    return env


class ExperimentTracker:
    """
    Writes JSON manifests (always) and drives an MLflow run (when available).

    Args:
        save_dir: Where checkpoints are written; manifests land beside them.
        config: Flat dict of hyperparameters. Recorded verbatim in every manifest
            and logged once as MLflow params.
        use_mlflow: Set False to skip MLflow entirely.
        experiment_name: MLflow experiment to group runs under.
        run_name: Defaults to a timestamp.
        tracking_dir: Where the local MLflow store lives (default: ./mlruns).
        resumed_from: Path of the checkpoint this run continues, if any.
    """

    def __init__(self, save_dir, config, use_mlflow=True,
                 experiment_name=DEFAULT_EXPERIMENT_NAME, run_name=None,
                 tracking_dir='mlruns', resumed_from=None):
        self.save_dir = save_dir
        self.config = dict(config)
        self.environment = collect_environment()
        self.run_name = run_name or datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.resumed_from = resumed_from
        self._mlflow = None
        self._warned = False

        if use_mlflow:
            self._start_mlflow(experiment_name, tracking_dir)

    # ------------------------------------------------------------------ MLflow

    @staticmethod
    def _resolve_tracking_uri(tracking_dir):
        """
        Pick a tracking backend.

        MLflow >= 3 refuses the plain filesystem store ('./mlruns') unless
        MLFLOW_ALLOW_FILE_STORE is set, so a local SQLite database is the
        default. An explicit MLFLOW_TRACKING_URI always wins.

        Returns (tracking_uri, artifact_uri, hint_command).
        """
        env_uri = os.environ.get('MLFLOW_TRACKING_URI')
        root = Path(tracking_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        artifact_uri = (root / 'artifacts').as_uri()

        if env_uri:
            return env_uri, artifact_uri, f'mlflow ui --backend-store-uri {env_uri}'

        db_uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
        return db_uri, artifact_uri, f'mlflow ui --backend-store-uri {db_uri}'

    def _start_mlflow(self, experiment_name, tracking_dir):
        # MLflow probes for git metadata through GitPython. With no git on PATH
        # that prints three multi-paragraph warnings on every single run. We
        # already capture the commit ourselves in collect_environment(), which
        # degrades to None silently, so this metadata is redundant anyway.
        #
        # Two independent guards, because they fail differently:
        #   - the env var stops GitPython raising ImportError at import time
        #     (setdefault, so an explicit user setting still wins)
        #   - silencing mlflow.utils.git_utils suppresses the warning even when
        #     the import still fails, e.g. if `git` got imported before this ran
        os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')
        logging.getLogger('mlflow.utils.git_utils').setLevel(logging.ERROR)
        try:
            import mlflow
        except ImportError:
            print("MLflow not installed - experiment tracking disabled. "
                  "Run 'pip install mlflow' to enable it. "
                  "JSON manifests are written either way.")
            return

        try:
            tracking_uri, artifact_uri, hint = self._resolve_tracking_uri(tracking_dir)
            mlflow.set_tracking_uri(tracking_uri)

            # Create with an explicit artifact location the first time round;
            # set_experiment alone would not pin it.
            if mlflow.get_experiment_by_name(experiment_name) is None:
                mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=self.run_name)

            # Params are immutable per run, so they go in once, up front.
            mlflow.log_params({k: v for k, v in self.config.items() if v is not None})
            mlflow.set_tags({
                'loss_semantics_version': LOSS_SEMANTICS_VERSION,
                'resumed_from': self.resumed_from or 'scratch',
                **{f'env.{k}': v for k, v in self.environment.items() if v is not None},
            })

            self._mlflow = mlflow
            print(f"MLflow tracking active: experiment='{experiment_name}', "
                  f"run='{self.run_name}'")
            print(f"  View with:  {hint}")
        except Exception as e:
            print(f"MLflow init failed, continuing without it: {e}")
            self._mlflow = None

    def _warn_once(self, e):
        if not self._warned:
            print(f"MLflow logging failed, continuing without it: {e}")
            self._warned = True
        self._mlflow = None

    def log_epoch(self, epoch, train_metrics, val_metrics, learning_rates):
        """Record one epoch. `epoch` is 0-based and used as the MLflow step."""
        if self._mlflow is None:
            return
        payload = {}
        payload.update({f'train_{k}': float(v) for k, v in train_metrics.items()})
        payload.update({f'val_{k}': float(v) for k, v in val_metrics.items()})
        payload.update({f'lr_{k}': float(v) for k, v in learning_rates.items()})
        try:
            self._mlflow.log_metrics(payload, step=epoch)
        except Exception as e:
            self._warn_once(e)

    # ---------------------------------------------------------------- manifest

    def write_manifest(self, checkpoint_path, epoch, train_metrics, val_metrics,
                       learning_rates, best_val_loss, best_val_acc,
                       epochs_no_improve, architecture=None):
        """
        Write <checkpoint>.json describing exactly how this checkpoint was made.

        Returns the manifest path.
        """
        manifest = {
            'manifest_schema_version': MANIFEST_SCHEMA_VERSION,
            'loss_semantics_version': LOSS_SEMANTICS_VERSION,
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'run_name': self.run_name,
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': epoch,
            'epoch_display': epoch + 1,
            'resumed_from': self.resumed_from,
            'architecture': architecture or {},
            'config': self.config,
            'metrics': {
                'train': {k: float(v) for k, v in train_metrics.items()},
                'val': {k: float(v) for k, v in val_metrics.items()},
                'learning_rates': {k: float(v) for k, v in learning_rates.items()},
                'best_val_loss': float(best_val_loss),
                'best_val_acc': float(best_val_acc),
                'epochs_no_improve': int(epochs_no_improve),
            },
            'environment': self.environment,
        }

        manifest_path = os.path.splitext(checkpoint_path)[0] + '.json'
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to write manifest {manifest_path}: {e}")
            return None

        if self._mlflow is not None:
            try:
                self._mlflow.log_artifact(manifest_path, artifact_path='manifests')
            except Exception as e:
                self._warn_once(e)

        return manifest_path

    # ------------------------------------------------------------------- close

    def close(self, best_val_loss=None, best_val_acc=None,
              final_checkpoint=None, log_model_artifact=False):
        """
        Finish the run.

        log_model_artifact copies the .pth into the MLflow store. Off by default:
        these checkpoints carry optimizer state and run to ~100 MB+.
        """
        if self._mlflow is None:
            return
        try:
            summary = {}
            if best_val_loss is not None:
                summary['best_val_loss'] = float(best_val_loss)
            if best_val_acc is not None:
                summary['best_val_acc'] = float(best_val_acc)
            if summary:
                self._mlflow.log_metrics(summary)
            if log_model_artifact and final_checkpoint and os.path.exists(final_checkpoint):
                self._mlflow.log_artifact(final_checkpoint, artifact_path='model')
            self._mlflow.end_run()
            print("MLflow run closed.")
        except Exception as e:
            print(f"MLflow shutdown failed: {e}")
        finally:
            self._mlflow = None


class _TeeStream:
    """
    Write to a stream and a file at once.

    isatty() delegates to the wrapped stream so tqdm's `disable=None` still sees
    the real terminal state and does not start drawing progress bars into a file.
    """

    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_file_logging(save_dir, filename='train.log'):
    """
    Mirror stdout into <save_dir>/<filename> so a background run always leaves a
    log behind without needing a shell redirect.

    Only stdout is teed. tqdm draws on stderr, which is deliberately left alone -
    that keeps the in-place progress-bar redraws out of the log file while still
    showing them on an interactive terminal.

    Do NOT also redirect the shell to this same path, or two handles will
    interleave into one file. Returns the log path.
    """
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)

    # line buffered, append so a resumed run keeps the earlier history
    handle = open(log_path, 'a', encoding='utf-8', buffering=1)
    handle.write(f"\n{'=' * 78}\nsession started {datetime.now().astimezone().isoformat(timespec='seconds')}\n{'=' * 78}\n")
    sys.stdout = _TeeStream(sys.stdout, handle)
    return log_path


def read_manifest(checkpoint_path):
    """
    Load the manifest belonging to a checkpoint, or None if there isn't one.

    Use this instead of guessing a checkpoint's provenance from its structure.
    """
    manifest_path = os.path.splitext(checkpoint_path)[0] + '.json'
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def summarize_checkpoints(directory):
    """
    One-line summary per checkpoint in `directory`, newest epoch first.

    Checkpoints without a manifest predate this tracking, and their losses are
    NOT comparable with loss_semantics_version 2 numbers.
    """
    rows = []
    if not os.path.isdir(directory):
        return rows
    for name in sorted(os.listdir(directory)):
        if not name.endswith('.pth'):
            continue
        path = os.path.join(directory, name)
        m = read_manifest(path)
        if m is None:
            rows.append({'checkpoint': name, 'manifest': False})
            continue
        rows.append({
            'checkpoint': name,
            'manifest': True,
            'epoch': m.get('epoch_display'),
            'loss_semantics_version': m.get('loss_semantics_version'),
            'val_loss': m.get('metrics', {}).get('val', {}).get('loss'),
            'recall_iou05': m.get('metrics', {}).get('val', {}).get('recall_iou05'),
            'created_at': m.get('created_at'),
            'run_name': m.get('run_name'),
        })
    rows.sort(key=lambda r: (r.get('epoch') or 0), reverse=True)
    return rows


if __name__ == '__main__':
    # `python tracking.py [dir]` prints what is sitting in a checkpoint folder.
    target = sys.argv[1] if len(sys.argv) > 1 else 'runs'
    rows = summarize_checkpoints(target)
    if not rows:
        print(f"No checkpoints found in {target}/")
        sys.exit(0)

    print(f"\nCheckpoints in {target}/\n" + "=" * 100)
    for r in rows:
        if not r['manifest']:
            print(f"{r['checkpoint']:<45} (no manifest - predates tracking, "
                  f"losses not comparable)")
            continue
        val_loss = r['val_loss']
        recall = r['recall_iou05']
        print(f"{r['checkpoint']:<45} epoch={r['epoch']:<4} "
              f"val_loss={val_loss:.4f} " if val_loss is not None else "",
              end='')
        print(f"recall={recall * 100:.2f}% " if recall is not None else "", end='')
        print(f"lsv={r['loss_semantics_version']} {r['created_at']}")
    print()
