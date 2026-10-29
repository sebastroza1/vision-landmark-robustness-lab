# Model Card

## Linear baseline

Logistic regression receives standardized engineered and normalized-coordinate features. It is a transparent reference for whether a nonlinear model adds value.

## PyTorch MLP

The MLP uses 58 inputs, hidden layers of 48 and 16 ReLU units, 0.10 dropout, and one binary logit. AdamW uses learning rate 0.002 and weight decay 0.0001. Quick mode allows up to 35 epochs with early stopping after eight validation macro-F1 non-improvements. The checkpoint records feature names, class names, seed, and the explicit `synthetic_only` flag.

The verified quick run completed 20 epochs. On the clean synthetic test set the linear baseline reached accuracy 0.942 and macro-F1 0.9419; the MLP reached accuracy 0.993 and macro-F1 0.9930. Both failed sharply under local corruption and missingness; see `RESULTS.md` rather than treating clean performance as a deployment claim.

Both models use a fixed 0.5 decision threshold. They classify generator-defined geometry patterns only and must not be applied to people, images, symptoms, or care decisions.

