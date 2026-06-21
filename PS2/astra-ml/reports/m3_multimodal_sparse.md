# M3 — Sparse/Novel Event Multimodal Forecaster

## Architecture

- **Text encoder**: google/muril-base-cased with Trainable LoRA Adapters
- **Adapters**: Trainable Text Projection layer
- **Fusion head**: [256, 128]
- **Training**: Multi-task (closure BCE + duration MSE)

## Leave-One-Cause-Out Cross-Validation (Cold-Start Evaluation)

For each sparse cause below, the model was trained on ALL other causes and evaluated on the held-out cause — simulating a true cold-start scenario where the model has zero training examples of that event type.

| Cause        | N  | Pos | ROC-AUC | PR-AUC | Status |
| ------------ | -- | --- | ------- | ------ | ------ |
| vip_movement | 20 | 16  | 0.5312  | 0.8530 | ok     |
| protest      | 15 | 6   | 0.3704  | 0.4778 | ok     |
| procession   | 72 | 19  | 0.5631  | 0.3005 | ok     |
| public_event | 84 | 39  | 0.4496  | 0.4394 | ok     |

## Failure Cases & Known Limitations

- **Single-class test sets**: Some sparse causes may have all-positive or all-negative closure labels, making AUC undefined. This is reported as 'single_class'.
- **Cold-start ceiling**: With zero training examples of the held-out cause, the model can only rely on semantic similarity (text embeddings) and structured feature patterns — there is a fundamental accuracy ceiling here.
- **Small sample sizes**: vip_movement (n≈20), protest (n≈15), procession (n≈14) — confidence intervals on these AUC numbers are very wide.
- **LoRA fine-tuning**: Trainable LoRA adapter layers are inserted dynamically; the base encoder remains frozen to prevent catastrophic forgetting.
- **Multilingual text**: MuRIL handles Kannada+English well, but mixed-script text with emojis (present in the data) may not be tokenized optimally.
