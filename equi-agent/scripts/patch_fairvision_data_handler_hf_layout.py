from __future__ import annotations

import argparse
from pathlib import Path


HELPER = r'''

# --- HuggingFace FairVision flat-layout compatibility patch ---
_HF_FAIRVISION_META_CACHE = {}

def _hf_fairvision_meta(data_file):
    import os
    import pandas as pd
    disease_dir = os.path.dirname(os.path.dirname(data_file))
    lookup_path = os.path.join(disease_dir, "metadata_lookup.csv")
    cache_key = lookup_path
    if cache_key not in _HF_FAIRVISION_META_CACHE:
        if os.path.exists(lookup_path):
            frame = pd.read_csv(lookup_path)
            _HF_FAIRVISION_META_CACHE[cache_key] = frame.set_index("filename").to_dict("index")
        else:
            _HF_FAIRVISION_META_CACHE[cache_key] = {}
    return _HF_FAIRVISION_META_CACHE[cache_key].get(os.path.basename(data_file), {})

def _hf_item(raw_data, key, data_file, default=None):
    if key in raw_data.files:
        return raw_data[key].item()
    meta = _hf_fairvision_meta(data_file)
    return meta.get(key, default)

def _hf_int_item(raw_data, key, data_file, default=-1):
    value = _hf_item(raw_data, key, data_file, default)
    try:
        return int(value)
    except Exception:
        return default
# --- End HuggingFace FairVision compatibility patch ---
'''


REPLACEMENTS = {
    "raw_data['amd_condition'].item()": "_hf_item(raw_data, 'amd_condition', data_file)",
    "raw_data['dr_subtype'].item()": "_hf_item(raw_data, 'dr_subtype', data_file)",
    "raw_data['glaucoma'].item()": "_hf_item(raw_data, 'glaucoma', data_file)",
    "raw_data['race'].item()": "_hf_item(raw_data, 'race', data_file)",
    "raw_data['male'].item()": "_hf_int_item(raw_data, 'male', data_file)",
    "raw_data['hispanic'].item()": "_hf_int_item(raw_data, 'hispanic', data_file)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch FairVision-main/src/data_handler.py to read staged metadata_lookup.csv when NPZ keys are absent."
    )
    parser.add_argument("--fairvision-repo", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.fairvision_repo / "src" / "data_handler.py"
    text = path.read_text()
    if "_hf_fairvision_meta" not in text:
        marker = "from torchvision import transforms\n"
        if marker not in text:
            raise ValueError(f"Could not find insertion marker in {path}")
        text = text.replace(marker, marker + HELPER, 1)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched={path}")


if __name__ == "__main__":
    main()
