"""Visualize prepared medical QA datasets and save plots to disk."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medicalqa_finetuning.commands import main


if __name__ == "__main__":
    main(["visualize", *sys.argv[1:]])
