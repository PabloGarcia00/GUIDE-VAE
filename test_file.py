import torch
import numpy as np
import joblib
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any
from pathlib import Path

class PulseDataset(Dataset):
    def __init__(self, inputs: np.ndarray, condition_dict: Dict[str, np.ndarray]):
        self.inputs = torch.from_numpy(inputs).float()
        self.conditions = {}

        for key, val in condition_dict.items():
            if self._is_strictly_integer(val):
                self.conditions[key] = torch.from_numpy(val).long()
            else:
                # If non-integer, we drop the entire condition as requested
                print(f"⚠️ Dropping condition '{key}': contains non-integer data.")

    def _is_strictly_integer(self, array: np.ndarray) -> bool:
        if np.issubdtype(array.dtype, np.integer):
            return True
        if np.issubdtype(array.dtype, np.floating):
            # Check if values are logically integers (e.g. 1.0 vs 1.1)
            return np.array_equal(array, array.astype(int))
        return False

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "kwh": self.inputs[idx],
            "features": {k: v[idx] for k, v in self.conditions.items()}
        }

class PulseDataModule(pl.LightningDataModule):
    def __init__(self, prep_dir: str = "preprocessing", batch_size=32):
        super().__init__()
        self.prep_path = Path(prep_dir) / "pulse_preprocessed.joblib"
        self.batch_size = batch_size
        self.stats = {}

    def setup(self, stage: str = None):
        if not self.prep_path.exists():
            raise FileNotFoundError(f"Preprocessed file not found at {self.prep_path}")

        payload = joblib.load(self.prep_path)
        
        self.stats['mean'] = torch.from_numpy(payload['stats']['mean']).float()
        self.stats['std'] = torch.from_numpy(payload['stats']['std']).float()

        self.train_set = PulseDataset(payload['train_inputs'], payload['condition_set']['train'])
        self.val_set = PulseDataset(payload['val_inputs'], payload['condition_set']['val'])
        self.test_set = PulseDataset(payload['test_inputs'], payload['condition_set']['test'])

        self.zero_id = payload["zero_id"]
        self.shift = payload["shift"]
        self.log_space= payload["log_space"]


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def reconstruct_kwh(self, Y: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of zero_preserved_log_denormalize.
        """
        X = Y.clone()
        is_zero = (X == self.zero_id)
        
        # Guard against log(negative/zero) for values we will mask anyway
        X[is_zero] = 1.0 
        
        if self.log_space:
            X_log = X
        else:
            X_log = torch.log(X)
            
        X_log = (X_log - self.shift) * self.stats['std'] + self.stats['mean']
        X = torch.exp(X_log)
        
        X[is_zero] = 0.0
        return X
