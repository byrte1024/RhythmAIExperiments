# Experiment 35-B - Full-Band Mel Ramps (Nuclear)

## Hypothesis

Exp 35 showed edge-band ramps provide 5.0% context delta but the conv filters them out. The ramps need to be inescapable.

**Nuclear approach**: halve all audio energy, then add ramps to ALL 80 mel bands:
```
mel = mel * 0.5 + ramp * 10.0  (broadcast across all bands)
```

The ramp signal is uniform across all frequency bands. The conv stem cannot learn to separate ramps from audio — they're mixed at every frequency. The model must process both signals simultaneously.

### Changes from exp 35

- Ramps added to ALL 80 mel bands (was: only bands 0-2, 77-79)
- Audio scaled by 0.5 before ramp addition (was: no scaling)
- No per-band intensity fading (was: 100%/50%/25% fade)

### Risk

- Halving audio may degrade onset detection quality too much — the model loses 3dB of audio signal.
- The uniform ramp across all bands destroys frequency-specific information at event positions.
- May be too aggressive — could try 0.75 scaling or lower ramp amplitude if results are poor.


## Result

*Pending*

## Lesson

*Pending*
