"""Prepare medical QA datasets."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medicalqa_finetuning.commands import main


if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
