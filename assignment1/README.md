Assignment 1 Instructions:
Problem 1: Benchmark Stationarity Detection
Write a function named is_stationary(samples: list[float]) -> dict[str, Any] that determines whether benchmark execution times remained steady throughout a test run rather than drifting over time. The input list must preserve the exact chronological order in which measurements were captured.

Requirements:
1. Input Validation:
• If the input list contains fewer than MIN_SAMPLES_FOR_STATIONARITY (12) samples, return an unknown record stating that there are too few samples to divide into thirds.
• Compute the overall median of the entire sample list. If this median is less than or equal to 0, return an unknown record stating that the median is not positive.

2. Splitting and Segment Analysis:
• Let k = ⌊n/3⌋, where n is the total number of samples.
• Extract the first third (the first k items) and calculate its median.
• Extract the last third (the last k items) and calculate its median.

3. Drift Calculation:
• Calculate the raw drift (∆) as:
∆ = median(last third) − median(first third)
• Compute the relative drift by dividing the absolute value of ∆ by the overall median of the run:
relative_drift = |∆| / overall_median

4. Direction Classification:
• Identify the direction of drift as a string:
– "slower" if ∆ > 0
– "faster" if ∆ < 0
– "flat" if ∆ = 0

5. Output Record:
• Return a measured record where the primary boolean result is True if the relative drift is less than or equal to STATIONARITY_TOL (0.10, or 10%), and False otherwise.
• Include the metadata fields:
– first_third_median_ms: Median of the first third, rounded to 4 decimal places.
– last_third_median_ms: Median of the last third, rounded to 4 decimal places.
– drift_ms: Raw drift (∆), rounded to 4 decimal places.
– drift_relative: Relative drift, rounded to 4 decimal places.
– direction: The classified direction string.
– tolerance: The constant STATIONARITY_TOL.
