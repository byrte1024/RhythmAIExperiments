# Experiment 47-D - Binary STOP Head with Forward-Pool Gate

## Hypothesis

47-C showed the gate can't learn from the cursor token alone -- it's a 384-dim vector optimized for onset location, not onset existence. The STOP decision is fundamentally different: "is there ANY onset in the forward window?" requires looking at the whole forward region, not one point.

### Architecture change

Gate head now reads from mean-pooled forward tokens (126-249) with its own LayerNorm:

```
# onset head: cursor token 125 -> 500 logits (unchanged)
cursor = x[:, 125, :]
onset_logits = head_proj(head_norm(cursor))

# gate head: mean-pool forward tokens -> 1 logit (new)
forward_pool = x[:, 126:, :].mean(dim=1)
gate_logit = gate_head(gate_norm(forward_pool))
```

The gate sees a summary of all audio content ahead of the cursor. If the forward window is silent/empty, the pool will be "flat" and the gate learns to output STOP. If there's a strong onset somewhere ahead, the pool picks it up.

Same balanced focal BCE from 47-C, same gate_weight=2.0 (now the gradient signal reaches different parameters than onset head, so low weight may be fine).

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_47d --model-type event_embed --binary-stop --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
