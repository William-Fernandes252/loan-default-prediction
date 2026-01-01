import sys

import numpy as np
import pandas as pd


def generate_taiwan_credit(output_path: str, n_rows: int = 100):
    """Generates a synthetic Taiwan Credit dataset."""
    np.random.seed(42)

    data = {
        "ID": range(1, n_rows + 1),
        "LIMIT_BAL": np.random.randint(10000, 500000, n_rows),
        "SEX": np.random.choice([1, 2], n_rows),
        "EDUCATION": np.random.choice([1, 2, 3, 4], n_rows),
        "MARRIAGE": np.random.choice([1, 2, 3], n_rows),
        "AGE": np.random.randint(21, 60, n_rows),
        "default.payment.next.month": np.random.choice([0, 1], n_rows),
    }

    # Add Pay and Bill columns (PAY_0 to PAY_6, BILL_AMT1 to BILL_AMT6, etc.)
    # Note: Dataset usually skips index 1 for PAY (PAY_0, PAY_2, etc) or uses 0-6.
    # Based on your transformer regex r"PAY_[0-6]", we generate 0-6.
    for i in range(7):
        data[f"PAY_{i}"] = np.random.randint(-2, 9, n_rows)

    for i in range(1, 7):
        data[f"BILL_AMT{i}"] = np.random.randint(0, 100000, n_rows)
        data[f"PAY_AMT{i}"] = np.random.randint(0, 50000, n_rows)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_rows} rows at {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_dataset.py <output_path>")
        sys.exit(1)
    generate_taiwan_credit(sys.argv[1])
