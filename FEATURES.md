# Feature Definitions

Landmarks are centered by their available-point centroid. When both eye anchors exist, the eye line is rotated to horizontal and its distance becomes the scale unit. Missing anchors fall back to an RMS scale without rotation.

The feature vector contains:

- 24 normalized x/y coordinates for 12 points;
- for each of five bilateral pairs: mirrored x error, vertical gap, radial-distance gap, and horizontal span;
- absolute x-axis deviation for two center points;
- 12 missing-point indicators.

Missing coordinates are set to zero only after normalization, with indicators retained. This is an engineering fallback, not a claim that zero imputation is optimal for real landmarks.

