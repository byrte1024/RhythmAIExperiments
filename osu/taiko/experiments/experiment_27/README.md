# Experiment 27 - Full Dataset (No Subsample)

## Hypothesis

Experiments 25 and 26 both plateau at ~69% HIT with subsample=4 (25% of training data). Exp 26 proved augmentation delays overfitting but doesn't raise the ceiling. The model memorizes the available audio→onset mappings regardless of noise.

**Real data diversity as regularization.** Subsample=4 means the model sees the same 25% of samples every epoch. Going to subsample=1 gives 4x more unique training examples — fundamentally different from augmentation which adds variation to the same underlying samples. More unique cursor positions, more unique audio contexts, more unique gap histories. Harder to memorize.

### Changes from exp 26

**Architecture: identical.** Same unified fusion model (~19M params), same heavy audio augmentation from exp 26.

**Training change: subsample 4 → 1.** Full dataset (~600K samples vs ~150K). ~4x slower per epoch but 4x more unique data.

Everything else unchanged — lr=3e-4, batch=48 (reduced from 64 for memory), train from scratch, same augmentation. Evals per epoch set to 4 — each eval step sees ~1 subsample-4 epoch of training data.

### Expected outcomes

1. **Much slower epochs** — ~4x wall clock per epoch.
2. **Less overfitting** — 4x more unique samples should keep the train/val gap tighter for longer.
3. **Higher ceiling** — if the ~69% plateau was from exhausting the subsample's diversity, the full dataset should push past it.
4. **Better context contribution** — more diverse gap histories may give the model more reason to use context features.

### Risk

- 4x slower iteration means fewer epochs in the same wall-clock time. If the model needs the same number of epochs to converge, we may not see the payoff.
- The ~69% ceiling may be fundamental to the task/architecture, not data-limited. In that case we burn 4x compute for the same result.
- Memory/throughput issues with larger dataset.

## Result

**Broke the ~69% HIT ceiling for the first time, but plateau and overfitting still occur.** Killed after eval 10 (~3.5 epochs).

| eval | epoch | HIT | Miss | Score | Acc | Frame err | Stop F1 | Train loss | Val loss | Ctx Δ |
|------|-------|-----|------|-------|-----|-----------|---------|------------|----------|-------|
| 1 | 1.25 | 66.5% | 32.9% | 0.304 | 47.8% | 14.3 | 0.440 | 3.521 | 2.701 | 8.1% |
| 2 | 1.50 | 67.5% | 32.0% | 0.317 | 49.4% | 12.7 | 0.491 | 3.376 | 2.635 | 2.3% |
| 3 | 1.75 | 68.7% | 30.9% | 0.329 | 50.7% | 12.3 | 0.517 | 3.290 | 2.593 | 1.6% |
| 4 | 1.00 | 68.6% | 31.0% | 0.327 | 50.5% | 12.5 | 0.487 | 3.228 | 2.597 | 2.4% |
| 5 | 2.25 | 69.4% | 30.2% | 0.338 | 51.5% | 11.6 | 0.530 | 2.989 | 2.566 | 2.0% |
| 6 | 2.50 | 69.0% | 30.6% | 0.333 | 51.2% | 12.0 | 0.521 | 2.969 | 2.571 | 1.3% |
| 7 | 2.75 | 69.2% | 30.3% | 0.335 | 51.3% | 12.1 | 0.522 | 2.950 | 2.570 | 1.4% |
| **8** | **2.00** | **69.8%** | **29.8%** | **0.343** | **51.5%** | **11.5** | 0.525 | 2.935 | **2.560** | 1.5% |
| 9 | 3.25 | 69.4% | 30.2% | 0.337 | 51.3% | 11.9 | **0.535** | 2.866 | 2.580 | 1.2% |
| 10 | 3.50 | 69.2% | 30.4% | 0.335 | 51.4% | 12.4 | 0.494 | 2.856 | 2.590 | 1.6% |

**What worked:**
- **New all-time highs.** 69.8% HIT (eval 8), first time breaking the ~69% ceiling that held across exp 14, 25, and 26. Also: 29.8% miss (first below 30%), 0.343 score, 51.5% accuracy, 2.560 val loss.
- **Slower overfitting.** Val loss dropped until eval 8 (epoch 2.0) before rising — exp 26 peaked at eval 4 (epoch 4 with subsample=4, same effective training). The full dataset provided ~2 extra useful epochs of training.
- **Faster convergence per wall-clock.** Reached exp 26 E7-level performance (68.8% HIT) by eval 3 (~0.75 epochs), because each mini-batch contains unique samples rather than repeated subsampled data.

**What didn't work:**
- **Plateau still occurred.** HIT oscillated 69.0-69.8% from eval 5 onward. Val loss crept from 2.560 (eval 8) to 2.590 (eval 10) while train loss kept falling. Same overfitting dynamic as exp 25/26, just delayed.
- **Context contribution still collapsed.** 8.1% (eval 1) → 1.2-2.4% (eval 4+). More diverse gap histories did not prevent the model from converging to audio dominance.
- **Ceiling raised only ~1pp.** 69.8% vs 68.9% (exp 14) / 68.8% (exp 26). Meaningful but not transformative.

**Qualitative analysis (from inference and graphs):**

The model's errors are systematic, not random:
- **Over-prediction (shyness).** The 2.0x ratio band dominates the ratio confusion chart. The model skips notes rather than adding extra ones. For a pattern like `150 150 75 75 150`, it generates `150 150 150 150` — defaulting to longer gaps.
- **Top-10 contains the answer 96% of the time.** The model narrows candidates to a small set of rhythmically valid options but can't pick the right one. Pattern disambiguation (75 vs 150 when both fit the audio) is the core failure mode.
- **Entropy correlates with target distance.** Short gaps: confident and accurate (cyan clump, bottom-left of entropy heatmap). Long gaps: high entropy, many options considered, often wrong.
- **Almost no noise in predictions.** Forward error graph shows predictions cluster at meaningful ratios (1.0x, 0.5x, 2.0x). The model learned rhythmic validity but not pattern specificity.

**Comparison across experiments 14, 25, 26, 27:**

| | Exp 14 | Exp 25 | Exp 26 | Exp 27 |
|---|---|---|---|---|
| Architecture | Audio+cross-attn | Unified fusion | Unified fusion | Unified fusion |
| Augmentation | Standard | Light | Heavy | Heavy |
| Subsample | 4 | 4 | 4 | 1 (full) |
| Best HIT | 68.9% (E8) | 68.6% (E5) | 68.8% (E7) | **69.8%** (eval 8) |
| Best val loss | ~2.65 | 2.623 | 2.627 | **2.560** |
| Overfitting onset | E4 | E2 | E5 | eval 8 (~E2) |
| Final ctx delta | ~0% | 2.3% | 1.7% | 1.5% |

## Graphs

Best eval (eval 8, epoch 2.0):

![Prediction Distribution](epoch_008_pred_dist.png)
![Scatter](epoch_008_scatter.png)
![Heatmap](epoch_008_heatmap.png)
![Ratio Scatter](epoch_008_ratio_scatter.png)
![Ratio Heatmap](epoch_008_ratio_heatmap.png)
![Ratio Confusion](epoch_008_ratio_confusion.png)
![Forward Error Scatter](epoch_008_forward_error_scatter.png)
![Forward Error Heatmap](epoch_008_forward_error_heatmap.png)
![Frame vs Ratio Scatter](epoch_008_frame_vs_ratio_scatter.png)
![Frame vs Ratio Heatmap](epoch_008_frame_vs_ratio_heatmap.png)
![Ratio in Density Scatter](epoch_008_ratio_in_density_scatter.png)
![Ratio in Density Heatmap](epoch_008_ratio_in_density_heatmap.png)
![Entropy Heatmap](epoch_008_entropy_heatmap.png)
![Entropy Hit vs Miss](epoch_008_entropy_hit_vs_miss.png)
![Top-K Accuracy](epoch_008_topk_accuracy.png)
![Top-K Quality](epoch_008_topk_quality.png)
![Accuracy by Context](epoch_008_accuracy_by_context.png)

## Lesson

- **Data diversity raises the ceiling, but only modestly.** 4x more unique data → ~1pp improvement (68.9% → 69.8%). Diminishing returns — the bottleneck is not data volume.
- **Overfitting is delayed, not eliminated.** More data pushes the overfitting onset later but the same dynamic eventually plays out. The model exhausts what it can learn from audio and starts memorizing.
- **The ~70% ceiling is architectural, not data-limited.** Three experiments varying augmentation (25 vs 26) and data volume (26 vs 27) converge to the same range. The next improvement must come from how the model uses information, not how much information it sees.
- **Pattern disambiguation is the key remaining problem.** The model predicts rhythmically valid candidates (96% top-10) but can't distinguish between them. This is exactly what context should solve but doesn't. Hyperparameter tuning (focal loss, loss asymmetry) or architectural changes (ratio-space prediction, learnable cursor) are the next levers.
- **Full dataset is worth keeping** for future experiments — it consistently outperforms subsample=4 and the slower epochs are offset by faster convergence per eval.
