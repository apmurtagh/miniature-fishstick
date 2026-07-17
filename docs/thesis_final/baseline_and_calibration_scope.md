# Baseline and Calibration Scope

## Purpose

This note clarifies that the LightGBM model is a sufficient experimental testbed for EO-to-narrative governance evaluation, not a claim of state-of-the-art IEEE-CIS performance.

## Baseline Metrics

The final baseline reports:

- ROC-AUC: 0.8687;
- PR-AUC: 0.4594;
- Brier score: 0.0236;
- ECE, 10-bin: 0.0043.

## Interpretation

These metrics show that the baseline is strong enough to support downstream explanation experiments. The thesis contribution is the governance-controllable EO-to-narrative layer, not model leaderboard optimisation.

## Calibration Caveat

ECE can be sensitive to binning choices, class imbalance and probability concentration. The reported 10-bin ECE should therefore be interpreted as an offline calibration diagnostic for the experimental testbed rather than proof of production calibration.

## Thesis-Safe Claim

The baseline is an adequate and calibrated-enough decision substrate for evaluating explanation governance. The thesis does not claim production model approval, portfolio transferability or state-of-the-art IEEE-CIS performance.
