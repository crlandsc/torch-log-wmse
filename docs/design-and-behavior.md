# How logWMSE works and how it behaves

This is the conceptual tour: what the metric is actually measuring, why it takes three inputs instead of the usual two, and the handful of behaviours that surprise people. You do not need any of it to use the library, but it helps to know what the number means and where its edges are.

## What it measures

At heart logWMSE is a mean squared error — the average squared difference between an estimate and a target — with two changes that make it behave well for audio.

First, the error is **frequency-weighted** before it is measured: differences in frequency bands the ear cares about count for more, and differences in bands the ear barely hears count for much less. Second, the result is put through a **logarithm** and scaled, which does two things at once. It turns MSE's awkward range (values crammed between about 1e-8 and 1e-3) into a readable one that looks like SI-SDR, and it matches the fact that human loudness perception is itself logarithmic. So a logWMSE score is, in plain terms, "how loud is the leftover error, weighted by how much the ear would care about it, on a log scale."

## The frequency weighting

The weighting is a fixed filter shaped like human hearing sensitivity. It rolls off the extremes — deep bass is discounted by roughly 24 dB at 30 Hz, and the very top by about 31 dB at 20 kHz — and gently emphasises the presence region around 3 kHz where we are most sensitive. Across the audible range it spans about 35 dB.

The point is to stop the metric from rewarding a model for fixing errors nobody can hear. An error buried at 30 Hz contributes far less to the score than the same-sized error at 3 kHz, which is usually what you want from a *perceptual* objective. It is also the reason logWMSE is not an SDR proxy: SDR counts every frequency equally, so the two will disagree, especially on low-frequency content. (See the [training guide](training-guide.md) for what that means when you use it as a loss.)

## Why three inputs

Most audio losses take two things: the estimate and the target. logWMSE takes three — it also wants the original **unprocessed mixture**.

That third input is what makes digital silence work. If a stem's target is pure silence, there is nothing to normalise against in the usual way, and metrics like SI-SDR simply break. By keeping the mixture around, logWMSE can measure how much of the original signal the model *removed*, which is well-defined even when the answer is "all of it". The mixture is the reference level the error is measured against.

It also gives the metric its scale invariance: because the error is judged relative to the mixture, gaining all three inputs by the same amount leaves the score unchanged. This is invariance to **joint** gain — turning everything up or down together. It is a different thing from scaling only the estimate, which the metric does penalise (see Limitations).

## Scale invariance, and where it stops

Joint-gain invariance holds until the mixture gets *extremely* quiet — around a root-mean-square level of 1e-8, which is far below anything normalised audio ever reaches. Below that, a numerical floor engages to keep the math well-defined, and the score starts to drift. The exact point depends a little on the spectral content of the mixture, because the floor acts on the frequency-weighted energy. In practice this only matters for pathologically attenuated signal; normalised audio is nowhere near it.

## Other sample rates

The metric is designed at 44.1 kHz, and that is where it is meant to live. It still runs at other rates, but it is worth understanding what happens, because scores at other rates are **not comparable** to 44.1 kHz scores.

The original numpy implementation handled other rates by resampling the *audio* up to 44.1 kHz. This implementation does the opposite: it resamples the *weighting filter* down to your audio's rate. That is a deliberate choice — it avoids touching your audio — but it means the two implementations diverge at other rates, and the divergence is largest near the top of the band. As a concrete example, at 16 kHz a strong error just below the Nyquist frequency can shift the score by tens of units between the two implementations, because the original's resampling filters that error away before measuring it while this one still hears it. Neither is "wrong"; they are measuring slightly different things once you leave 44.1 kHz. The safe rule: treat sub-44.1 kHz scores as internally consistent, and do not compare them across implementations or across sample rates.

## Comparing scores across models

A couple of things to keep in mind when you put logWMSE numbers side by side.

- **Per-stem values are the portable ones.** The per-stem scores (`per_stem(...)`) match the original numpy implementation at 44.1 kHz whatever the `p` setting is. The single pooled number matches it too at the default `p = 0`; at other `p` it deliberately does not, for multi-stem or stereo inputs.
- **Stem count inflates the score.** Perfect or near-silent stems pull the aggregate up, so a 4-stem model and a 16-stem model are not directly comparable on their pooled score. Compare like with like, or compare per-stem.

## Limitations, in a little more depth

- **Perceptual, not signal-fidelity.** Because it discounts inaudible bands, training against logWMSE will generally cost you SDR relative to an unweighted loss, concentrated in the low-frequency stems. That trade is the whole point of the metric. If SDR is your actual target, use an SDR-matched loss.
- **Only joint gain is free.** Scaling, inverting the polarity of, or adding an offset to the *estimate alone* all change the score. Unlike SI-SDR, logWMSE does not solve for an optimal estimate scale, so an estimate that is right apart from a gain error is penalised for that gain error.
- **It is inspired by hearing, not a model of it.** The frequency weighting captures sensitivity, but not the richer parts of auditory perception — auditory masking, for instance, is not modelled.
