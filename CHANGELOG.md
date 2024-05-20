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

## 0.1.5 (2024-05-20)

#### Convolution Bug
The convolution operation was previously introducing an unintended time shift due to incorrect padding and trimming. This was causing models to inadvertently learn these time shifts when the operation was used as a loss function. This issue has now been corrected. The convolution operation is now time-invariant, meaning it will not introduce any unwanted time shifts.