# Conditional component evaluation of supported supervision

Registered before any all-TRAIN fit or component-validation predictions. Execute
only after the full TRAIN proposal audit completes and passes its preregistered
joint/town/phase non-regression and background-reduction criteria, with exact
48,865 sampled-row parity and independent metric recount. If that audit fails,
this protocol remains unexecuted. No candidate choice uses component validation.

Fit exactly one supported joint100 model on all49,088 retained original TRAIN rows
and their two existing gamma variants (147,264 fitting rows). Use the exact same
87RGBfeatures, existing four-class labels and fold-study settings: seed42,
learning_rate.1,100iterations×4trees,15leaves,min_samples_leaf50,L2=10,max_bins255,
no early stopping. Fit weights: total background mass1/2, each phase1/6. No other
hyperparameters, support radius, threshold or representation may be selected here.
The2,006 unsupported positives remain excluded from loss only; original caches,
labels and archived failures remain unchanged. Save all-TRAIN and component-VAL
probabilities and exact pickle serialization parity.

Evaluate the resulting head ONCE on the original38,543proposal rows from144images
with72housing labels. No GT-only anchors, row removal, relabeling, threshold sweep,
extra town data or augmented validation. Accept sum foreground≥.5, predict the
largest foreground phase probability, use the unchanged score ordering and
one-to-one point-in-housing matching. Keep all rejected proposals' probabilities.

The original component gate remains: ≥44correct housing-and-phase detections,
no town or phase regression versus the original bulb parent, and ≥90% reduction
in known-background detections versus its37,241. All unknown proposals are counted
separately, not converted to correct negatives. Even a component pass is NOT a
camera-driving qualification. This repeatedly inspected component set is development
evidence, not an untouched test. No native deployment or finalTown05 run follows
automatically; a point detector still requires a separately verified runtime adapter.

Output outputs/navigation_signal_supported_final_20261022. One CPU fit,2threads,
180s child,900s stage,100MB artifacts; hash reference existing caches. Retain exact
sources/model/protocol/candidate-selection provenance; STOP/normal owned cleanup,
no retries. Same original14GBnet budget and10GBminimum guardian.
