# Using logWMSE as a loss

`LogWMSELoss` is the negated metric: lower is better, and you can minimise it directly. It trains audio separation and denoising models, and its main draw over SDR-style losses is that it handles **digital-silence targets** — a stem that is supposed to be completely silent — without blowing up.

This guide covers the handful of things that actually change how training goes. None of it is required reading to get started (the loss works out of the box), but a few of its behaviours are different enough from a plain MSE loss to be worth understanding.

## The score, and its ceiling

The loss is scaled to sit in a readable range, roughly like SI-SDR: think of the metric as "3 is poor, 30 is very good", and the loss as its negative. Because the loss is a logarithm, it has a hard ceiling on the metric side: a perfect estimate — or an all-silent triplet where nothing needed separating — scores exactly **+73.6827**, and the loss bottoms out at −73.6827. A value pinned there means "no measurable error", not a bug.

## How the gradient behaves

This is the one property most worth internalising, because it is the opposite of what MSE does.

A plain MSE loss has a gradient that **shrinks** as the estimate gets closer to the target — near the optimum it goes flat, and learning slows down. logWMSE does the reverse: because it takes the logarithm of the error, the gradient **grows** as the estimate improves. The slope of `log(x)` is `1/x`, so the smaller the remaining error, the harder the loss pushes. This is the same behaviour SI-SDR has, and it is generally a good thing — the model keeps getting a strong signal to polish a stem that is already close.

The practical consequence is that gradients can get large late in training, so you should **use gradient clipping**. Two things are worth knowing about that:

- Clipping at a normal threshold (say a max norm of 1) will be inactive early, when the model is untrained and gradients are small, and then active for most of training once the estimate gets good. Once it kicks in it rescales every step to the same size, which effectively turns the loss into a normalised-gradient method. That is usually fine — just know it is happening, so you are not surprised that your gradient-norm plot flattens.
- The growth is not endless. The gradient rises steeply as a stem converges (roughly a factor of 900 between a residual 10 dB down and one 70 dB down), peaks around 80 dB down, and then falls off again. So the gradient norm alone does not tell you how converged a stem is — two very different levels can produce the same norm.

In practice, separation is hard enough that a model rarely converges far enough to reach that peak, so for most training runs you are on the gentle, well-behaved part of the curve. This is the same reason SNR-style losses do not blow up in practice even though they can in theory.

If late-training gradient growth does become a problem, the `p` knob below can flatten it.

## Combining stems: the `p` knob

A multi-stem model produces one error per stem, and those have to become a single number. The `p` argument controls how they combine.

- **`p = 0` (the default)** is the mean of the per-stem logs — the exact aggregation every version before 1.0.0 used. Multi-stem and stereo scores stay comparable with previously published logWMSE numbers, so if you do nothing, nothing about aggregation changes.
- **`p = 0.5`** is the one alternative worth knowing about. It spreads the gradient more evenly across stems as they converge, which flattens the late-training growth described above. Reach for it only if that growth actually causes you trouble; otherwise leave `p` alone.

One honest caveat if you go down this road: `p = 0.5` does **not** perfectly equalise the gradient across stems in general. It evens out the part that depends on each stem's error level, but not the part that depends on *where in the frequency spectrum* a stem's error sits. Two stems with the same amount of error but energy in very different bands can still get very different gradient, at any `p`, because the frequency weighting spans about 35 dB across the audible range. `p` is a useful dial, not a guarantee.

Single-stem mono models are unaffected by `p` entirely — with one value to combine, every setting agrees.

Neither default has been validated against SDR on a real separation run yet; `p = 0` is chosen because it is the historically comparable one and makes no unproven claim.

## Mixed precision (AMP)

The frequency weighting runs a Fourier transform, and `torch.fft` has no half-precision kernel on CPU or on Apple Silicon (MPS). That sounds like it should rule out mixed precision, but it does not — you just use it the normal way.

**Use `torch.autocast`.** Inside an autocast region the transform's inputs are handled in float32 and the filtered path works at both float16 and bfloat16, returning the score in float32 at negligible cost. What matters is that the tensors reaching the transform end up float32. In a normal setup that happens for free: your model's estimate is half precision, but your targets and mixture come from the dataloader in float32, and subtracting a half-precision estimate from a float32 target promotes the result back to float32.

- **On CPU and MPS**, keep your targets and mixture in float32. In PyTorch Lightning that means `precision="bf16-mixed"`, not `"bf16-true"` — the "true" mode casts your whole data path to half precision, and then the transform has nothing to promote and raises `Unsupported dtype`.
- **Do not reach for `bypass_filter=True` as a precision fix.** It is not one. `bypass_filter` removes the frequency weighting entirely, which changes what you are optimising.
- CUDA has not been tested here (no device was available), so treat the CUDA autocast path as unverified until you confirm it on your hardware.

## Audio level and segment length

The metric's *value* is invariant to gaining all three inputs together, but its *gradient* is not: gaining everything by a factor `g` scales the gradient by `1/g`. So your effective learning rate quietly depends on your audio level, with no sign of it in the loss curve. **Normalise your audio.**

Segment length is a much smaller effect than it looks. The per-sample gradient scales with the crop length, but the gradient that actually reaches your model's parameters was nearly flat across a 16× range of crop lengths (within a few percent, on a fully-convolutional model). So cropping mostly does not change your effective learning rate — you do not need to retune it every time you change the clip length.

## A note on memory

The filtered path is heavier than a plain MSE loss — roughly 9× the peak memory for a single forward and backward pass — because it holds a complex spectrum and computes a separate transform for the mixture reference. At realistic batch sizes this is usually a small fraction of the model's own footprint, but it is worth knowing if you are swapping logWMSE in for MSE at a fixed batch size and suddenly find yourself tighter on memory.
