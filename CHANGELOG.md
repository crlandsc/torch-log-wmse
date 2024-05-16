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
