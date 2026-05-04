import pandas as pd
from pathlib import Path


def ingest_nasa_arc(input_file: Path, output_dir: Path):
    """
    Ingest NASA ARC data and save processed files.

    Args:
        input_file (Path): Path to the raw NASA ARC data file.
        output_dir (Path): Directory to save processed data.
    """
    data = pd.read_csv(input_file)

    # Example processing: filter and save
    processed_data = data.dropna(subset=["IR_ohm", "OCV_V"])
    output_file = output_dir / "processed_nasa_arc.csv"
    processed_data.to_csv(output_file, index=False)
    print(f"Processed NASA ARC data saved to {output_file}")


if __name__ == "__main__":
    raw_data_file = Path("data/raw/nasa_arc.csv")
    processed_data_dir = Path("data/processed")
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    ingest_nasa_arc(raw_data_file, processed_data_dir)