# Base vs Fine-Tuned Comparison

_Generated: 2026-06-06T07:06:46_

- Base model: `unsloth/Llama-3.2-1B`
- Fine-tuning: LoRA, 200 steps
- Eval samples per dataset: 50

| Dataset | Random | Base | Fine-tuned | Improvement |
|---------|-------:|-----:|-----------:|------------:|
| pubmedqa | 33.3% | 26.0% (13/50) | 64.0% (32/50) | +38.00% |

## Artifacts

### pubmedqa
- Adapter: `experiments/pubmedqa/final_adapter`
- Base eval: `experiments/pubmedqa/base_eval.csv`
- Fine-tuned eval: `experiments/pubmedqa/finetuned_eval.csv`
