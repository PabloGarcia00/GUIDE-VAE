import argparse
import json
import os
from pathlib import Path
import pickle

import gvae.preprocess_lib as preprocess_lib
import joblib

def load_config(json_file):
    with open(json_file, 'r') as file: return json.load(file)

def save_preprocessed_data(outputs, output_dir="preprocessed"):
    """
    Saves the return values of prepare_data_for_faraday to disk.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    (trainset, valset, testset, user_ids, 
     indices, condition_set, mean, std, log_space, shift, zero_id) = outputs

    # We save inputs separately from the condition_set for clarity
    data_payload = {
        "train_inputs": trainset.inputs,
        "val_inputs": valset.inputs,
        "test_inputs": testset.inputs,
        "condition_set": condition_set,
        "stats": {"mean": mean, "std": std},
        "user_ids": user_ids,
        "indices": indices,
        "log_space": log_space, 
        "shift": shift,
        "zero_id": zero_id,
    }
    
    joblib.dump(data_payload, path / "pulse_preprocessed.joblib")
    print(f"Successfully saved preprocessing data to {path}")

def get_preprocessed(config):
    outputs = preprocess_lib.prepare_data_for_faraday(config["data"])
    return outputs

def main(config_path, output_dir):
    config = load_config(config_path)
    outputs = get_preprocessed(config)
    save_preprocessed_data(outputs, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write preprocessed GUIDE-VAE data.")
    parser.add_argument("--config_path", type=str, default="c10_ldn.json", help="Path to the JSON configuration file.")
    parser.add_argument("--output_dir", type=str, default="preprocessed")
    args = parser.parse_args()
    main(args.config_path, args.output_dir)
