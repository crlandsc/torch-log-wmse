# Changelog

## 0.1.0 (2024-05-15)

Initial release

## 0.1.1 (2024-05-15)

#### Bug Fix
Removed device assignment that was causing errors on distributed training setups

## 0.1.2 (2024-05-16)

#### Bug Fix
Error in tensor shapes. Was processing as the incorrect [batch, channels, stem, samples] instead of the correct [batch, stem, channels, samples] in some places.

## 0.1.3 (2024-05-16)

#### Include Image
Moved frequency weighting image to be included in the package.

## 0.1.4 (2024-05-16)

#### Image Bug
Moving image again and correcting reference.

## 0.1.5 (2024-05-17)

#### Added Logo
Added repo logo.

## 0.1.6 (2024-05-20)

#### Convolution Bug
The convolution operation was previously introducing an unintended time shift due to incorrect padding and trimming. This was causing models to inadvertently learn these time shifts when the operation was used as a loss function. This issue has now been corrected. The convolution operation is now time-invariant, meaning it will not introduce any unwanted time shifts.

## 0.1.7 (2024-05-22)

#### RMS Bug
If the unprocessed audio was silent, a value was immediately being returned unrelated to the model predictions. While this is how the original metric was implemented [here](https://github.com/nomonosound/log-wmse-audio-quality), the RMS value should actually be replaced with epsilon so that the difference between a non-silent output and silent output can be measured.

Added test for silent input & target.

Adjusted circular shift to account for IR with odd number of samples.

## 0.1.8 (2024-05-22)

#### Minimum threshold fix
Changed scaling factor so RMS doesn't need to = 0, rather just be lower than the error threshold to replace with min value. If it only could be 0, then very tiny numbers even closer to 0 would still go through.

## 0.1.9 (2024-06-18)

#### Package name update
Publishing as `torch-log-wmse` (for brevity) as well as `torch-log-wmse-audio-quality`.

## 0.2.0 (2024-06-18)

#### Finalizing name change
Updated all references to **`torch_log_wmse`** independent of installation name (i.e. `pip install torch-log-wmse` or `pip install torch-log-wmse-audio-quality`).

Imports now **MUST** be done as the following:
```
from torch_log_wmse import LogWMSE
```

## 0.2.1 (2024-06-18)

#### Updated badges
Updated badge references in the `README`.

## 0.2.2 (2024-06-18)

#### Changed GitHub repo name & references
Updated GitHub repo name to appropriate `torch-log-wmse`. `torch-log-wmse-audio-quality` can still be installed, but everything will reference the new name (`torch-log-wmse`) moving forward.

## 0.2.3 (2024-06-19)

#### Weighted filter reference bug
Corrected file reference to pkl filter file.