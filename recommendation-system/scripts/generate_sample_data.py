from pathlib import Path

from src.config import get_settings
from src.data import create_sample_dataset


if __name__ == "__main__":
    path = create_sample_dataset(get_settings().processed_data_dir)
    print(path)

