import time
import torch
import os
import json


class ExperimentProfiler:
    """
    Tracks wall-clock time and GPU memory per experiment.
    Saved as profiling.json inside the experiment results folder
    so it loads alongside history.json automatically.
    """

    def __init__(self, experiment_name: str):
        self.experiment_name    = experiment_name
        self.epoch_times        = []
        self.peak_memory_mb     = 0.0
        self.start_time         = None
        self.total_time         = None
        self._epoch_start       = None
        self.trainable_params   = 0
        self.total_params       = 0
        self.checkpoint_size_mb = 0.0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.start_time = time.time()

    def start_epoch(self):
        self._epoch_start = time.time()

    def end_epoch(self):
        self.epoch_times.append(
            round(time.time() - self._epoch_start, 2)
        )
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e6
            self.peak_memory_mb = max(self.peak_memory_mb, peak)

    def end(self):
        self.total_time = round(time.time() - self.start_time, 1)

    def log_model(self, model):
        self.trainable_params = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
        self.total_params = sum(
            p.numel() for p in model.parameters()
        )

    def log_checkpoint_size(self, path: str):
        total = 0
        if os.path.isdir(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        elif os.path.isfile(path):
            total = os.path.getsize(path)
        self.checkpoint_size_mb = round(total / 1e6, 2)

    def summary(self) -> dict:
        avg = (sum(self.epoch_times) / len(self.epoch_times)
               if self.epoch_times else 0)
        pct = (round(100 * self.trainable_params / self.total_params, 2)
               if self.total_params > 0 else 0)
        return {
            "experiment":           self.experiment_name,
            "total_time_sec":       self.total_time,
            "total_time_human":     self._fmt(self.total_time or 0),
            "avg_epoch_time_sec":   round(avg, 1),
            "epoch_times_sec":      self.epoch_times,
            "peak_gpu_memory_mb":   round(self.peak_memory_mb, 1),
            "peak_gpu_memory_gb":   round(self.peak_memory_mb / 1024, 2),
            "trainable_params":     self.trainable_params,
            "total_params":         self.total_params,
            "trainable_pct":        pct,
            "checkpoint_size_mb":   self.checkpoint_size_mb,
        }

    def save(self, exp_path: str):
        """Save profiling.json inside experiment results folder."""
        os.makedirs(exp_path, exist_ok=True)
        with open(os.path.join(exp_path, "profiling.json"), "w") as f:
            json.dump(self.summary(), f, indent=2)

    def print_summary(self):
        s = self.summary()
        print(f"\n{'-'*50}")
        print(f"Profiling -- {s['experiment']}")
        print(f"  Total time:       {s['total_time_human']}")
        print(f"  Avg epoch:        {s['avg_epoch_time_sec']}s")
        print(f"  Peak GPU memory:  {s['peak_gpu_memory_gb']} GB")
        print(f"  Trainable params: {s['trainable_params']:,}")
        print(f"  Total params:     {s['total_params']:,}")
        print(f"  Trainable %:      {s['trainable_pct']}%")
        print(f"  Checkpoint size:  {s['checkpoint_size_mb']} MB")
        print(f"{'-'*50}")

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 60:   return f"{seconds:.0f}s"
        if seconds < 3600: return f"{seconds/60:.1f}min"
        return f"{seconds/3600:.1f}h"

    @staticmethod
    def load(exp_path: str):
        """
        Load profiling.json from an experiment results folder.
        Returns None if file does not exist.
        """
        path = os.path.join(exp_path, "profiling.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
