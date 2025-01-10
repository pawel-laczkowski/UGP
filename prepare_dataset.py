#!/usr/bin/env python3

import json
import logging
from pathlib import Path

from datasets import load_dataset

LOGGER = logging.getLogger(__name__)

DATASET_NAME = "dair-ai/emotion"

MAP_LABEL_TRANSLATION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


def save_limited_data(file_path: Path) -> None:
    all_text = file_path.read_text().split("\n")
    text = all_text
    save_path = file_path.parent / f"{file_path.stem}.json"
    save_path.write_text("\n".join(text))
    LOGGER.info(f"Saved encoded ({len(text)}) version in: {save_path}")


def save_as_translations(original_save_path: Path, data_to_save: list[dict]) -> None:
    file_name = "s2s-" + original_save_path.name
    file_path = original_save_path.parent / file_name

    LOGGER.info(f"Saving seq-to-seq version in: {file_path}")
    with open(file_path, "wt") as f_write:
        for data_line in data_to_save:
            label = data_line["label"]
            new_label = MAP_LABEL_TRANSLATION[label]
            data_line["label"] = new_label
            data_line_str = json.dumps(data_line)
            f_write.write(f"{data_line_str}\n")

    save_limited_data(file_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    loaded_data = load_dataset(DATASET_NAME)
    LOGGER.info(f"Loaded dataset {DATASET_NAME}: {loaded_data}")

    save_path = Path("data/")
    save_train_path = save_path / "train.json"
    save_valid_path = save_path / "valid.json"
    save_test_path = save_path / "test.json"
    if not save_path.exists():
        save_path.mkdir()

    # Read train and validation data
    data_train, data_valid, data_test = [], [], []
    for source_data, dataset in [
        (loaded_data["train"], data_train),
        (loaded_data["validation"], data_valid),
        (loaded_data["test"], data_test),
    ]:
        for i, data in enumerate(source_data):
            data_line = {
                "label": int(data["label"]),
                "text": data["text"],
            }
            dataset.append(data_line)
    LOGGER.info(f"Train: {len(data_train):6d}")
    LOGGER.info(f"Valid: {len(data_valid):6d}")
    LOGGER.info(f"Test : {len(data_test):6d}")

    # Save files
    for file_path, data_to_save in [
        (save_train_path, data_train),
        (save_valid_path, data_valid),
        (save_test_path, data_test),
    ]:
        LOGGER.info(f"Saving into: {file_path}")
        with open(file_path, "wt") as f_write:
            for data_line in data_to_save:
                data_line_str = json.dumps(data_line)
                f_write.write(f"{data_line_str}\n")

        save_limited_data(file_path)
        save_as_translations(file_path, data_to_save)


if __name__ == "__main__":
    main()
