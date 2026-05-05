import csv
from pathlib import Path


class ProgressLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "progress.csv"
        self.rows = []
        self.fieldnames = None

    def record(self, row):
        self.rows.append(row)
        self._write_csv()

    def _write_csv(self):
        fieldnames = sorted({key for row in self.rows for key in row.keys()})
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def maybe_plot(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipping progress plot.")
            return

        if not self.rows:
            return

        epochs = [row["epoch"] for row in self.rows]
        eval_returns = [row["eval_return_mean"] for row in self.rows]
        expl_returns = [row["expl_return_mean"] for row in self.rows]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, eval_returns, label="eval return")
        plt.plot(epochs, expl_returns, label="exploration return")
        plt.xlabel("epoch")
        plt.ylabel("return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "returns.png")
        plt.close()
