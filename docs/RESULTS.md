# Fine-Tuning Results

Base vs fine-tuned accuracy on held-out validation questions
(LoRA, 200 steps, 4-bit QLoRA on GPU, 50 eval samples per dataset).

![Base vs fine-tuned accuracy](../experiments/comparison.png)

| Dataset | Random | Base | Fine-tuned | Improvement |
|---------|-------:|-----:|-----------:|------------:|
| PubMedQA | 33.3% | 26% (13/50) | **64% (32/50)** | **+38%** |
| MedMCQA | 25% | 28% (14/50) | _pending (Colab)_ | — |

## Takeaways

- **Fine-tuning works:** PubMedQA accuracy went from below the random baseline
  (26%) to **64%** — more than doubling, with <1% of parameters trained.
- **Format is fixed too:** the base model rambled; the fine-tuned model answers
  cleanly (`Yes. ...` / `No. ...`), which is what makes the quiz explanations and
  the answer extractor reliable.
- **Train on GPU, not MPS:** local Apple MPS training diverged to NaN weights
  (0% accuracy). All training is done on a free Colab GPU — see
  [FINETUNING.md](FINETUNING.md).

## Reproduce

```bash
# Train on Colab GPU (one cell), then regenerate this plot locally:
python scripts/plot_comparison.py
```

Raw numbers: [`experiments/RESULTS.json`](../experiments/RESULTS.json) ·
per-question CSVs in `experiments/<dataset>/`.
