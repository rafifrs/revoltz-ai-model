from src.data_loader import load_all_cells
from src.feature_extractor import extract_snapshot_features
from src.model.model2_cell_matcher import CellMatcher
from src.config import DEFAULT_MODEL_PATH, DATA_DIR
import pandas as pd


def main():
    # Load all cells from the data directory
    cells = load_all_cells(DATA_DIR)

    # Extract snapshot features for each cell
    snapshots = []
    for cell in cells:
        snapshot = extract_snapshot_features(cell, at_cycle=100)
        if snapshot:
            snapshots.append(snapshot)

    # Create a DataFrame from the snapshots
    df = pd.DataFrame(snapshots).dropna(subset=['IR_ohm', 'OCV_V'])
    print(f"Total valid cells for training: {len(df)}")

    # Train the CellMatcher model
    model = CellMatcher(n_clusters=3)
    model.fit(df)

    # Save the trained model
    model.save(DEFAULT_MODEL_PATH)
    print(f"Model saved to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()