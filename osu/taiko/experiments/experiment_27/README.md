# Experiment 27 - Full Dataset (No Subsample)

## Hypothesis

Experiments 25 and 26 both plateau at ~69% HIT with subsample=4 (25% of training data). Exp 26 proved augmentation delays overfitting but doesn't raise the ceiling. The model memorizes the available audio→onset mappings regardless of noise.

**Real data diversity as regularization.** Subsample=4 means the model sees the same 25% of samples every epoch. Going to subsample=1 gives 4x more unique training examples — fundamentally different from augmentation which adds variation to the same underlying samples. More unique cursor positions, more unique audio contexts, more unique gap histories. Harder to memorize.

### Changes from exp 26

**Architecture: identical.** Same unified fusion model (~19M params), same heavy audio augmentation from exp 26.

**Training change: subsample 4 → 1.** Full dataset (~600K samples vs ~150K). ~4x slower per epoch but 4x more unique data.

Everything else unchanged — lr=3e-4, batch=64, train from scratch, same augmentation.

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

*Pending*

## Lesson

*Pending*
