# Fixed parent/latest detector-loss and GT-ROI audit

The expanded joint fit completed200 updates. Latest200 improves sparse detection
30→36/72, joint state24→26 and approximate backgrounds681→209. It passes protection
but remains unselected because detector loss.126094 exceeds parent.118660.
Preserve that selection; no further fitting, calibration, threshold change or
post-hoc promotion. This audit is descriptive, not a new selection rule.

Freeze exactly original selected50 and completed expanded-joint latest200. On
the unchanged144 component-development images, repeat the same FP32,batch2,
seed20261017 detector-loss calculation once per model. Record each of the four
loss components and reproduce saved totals within1e-6. No training/gradient or
optimizer update. All frozen features/normalization buffers must remain identical.

On the fixed72 positive validation GT boxes, extract shared frozen ROI features
and compare each head's binary foreground acceptance at.5. Report total/by-town
and individual margins. This is privileged GT-ROI diagnosis, not detector recall,
driving performance, a precision estimate or a source of fitting labels. No
negative threshold search and no alternate candidates. Frozen RPN metrics already
exist; do not repeat full RGB detection or change the retained component.

Fresh output/source hashes,300s cap, STOP and GPU lock. No policy, benchmark,
reward, heldout_dev/Town05 use or multi-hour work.
