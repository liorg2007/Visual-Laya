"""Data pipeline: JSONL training records, dataset/collate/mixing, augmentation and converters.

Heavy imports (datasets, PIL, images.py) happen lazily inside the functions that need them.
"""
from .schema import SPLITS, TrainRecord, n_options, normalize, read_jsonl, split_for, validate_record, write_jsonl

__all__ = ["SPLITS", "TrainRecord", "n_options", "normalize", "read_jsonl", "split_for", "validate_record",
           "write_jsonl"]
