# Navigation results —20October2026

> **Index updated through 23 October 2026; execution paused by the user.**
> No training/native owner is live. M6 final driving remains unrun. See the
> [detailed project report](journey/summary.md),
> [interview preparation](interview_preparation_20261023.md) and
> [pause record](navigation_pause_20261023.json). Historical result dates and
> original denominators below are retained.

Current objective remains the full M0–M6camera navigation project. Native driving,
rule understanding and predictive planning are separate evidence requirements.
No final Town05camera candidate is qualified yet.

| Evidence | Result | Report |
|---|---|---|
| Latest spatial-context audit | 1,228/1,248 joint and 3,221 background detections; fails all four original protections. No component follow-up or driving promotion. | [Full audit](navigation_signal_spatial_full_20261022.md) |
| Missing-positive coverage | 1,596 additional exact TRAIN bulb positives, 4,788 verified original/gamma feature vectors; prospective fit prepared but unstarted. | [Prospective protocol](navigation_signal_positive_protocol_20261022.md) |
| Expanded background supervision | TRAIN 1,232/1,248 correct; 4,848 background, down from 8,197 but still fails recall and background gates. No component follow-up. | [Independent recount and fixed gallery](navigation_signal_background_full_20261022.md) |
| Exact preprocessing acceleration | 236,284 TRAIN points: CPU features/probabilities exact; CUDA features exact with zero decision changes. Median measured GPU feature-plus-head path 22.87 ms/image, not full driving runtime. | [Timing, parity and limitations](navigation_signal_preprocessing_20261022.md) |
| Independent RGB plateau proposals | TRAIN1,234/1,248joint;8,197background fails fixed3,075.8limit. No component check. | [Full audit and false-positive gallery](navigation_signal_plateau_20261022.md) |
| Staged point interface / numeric-tree export | Dynamic current-frame ray tokens; GPUhead kernel4.32×faster with6.66e-16maxprobabilitydifference on4,096TRAINrows. Not integrated or full-pipeline speed. | [Runtime staging and limits](navigation_signal_runtime_staging_20261022.md) |
| RGB-only proposal refinement | Component43→55/72joint,127background; stillFAILSyellow12versus15. No newfit or deployment. | [Report and graph](navigation_signal_refinement_20261022.md) |
| Supported local signal supervision | TRAIN1,141→1,144/1,248; full background30,758→724. Component43/72joint/40background still FAILS yellow/Town10/total protection; no deployment. | [Report and graph](navigation_signal_supported_20261022.md) |
| TRAIN supervision geometry audit | 2,006/3,371 positives lack active pixels in detailed5×5;1,351 within red-repeat variation. No overlapping-box explanation; no benchmark edits. | [Audit and paired gallery](navigation_signal_support_geometry_20261022.md) |
| Familiar TRAINlong drive | ResNet1.376kmarrival,0contacts,0interventions;5minimum-speedinfractions. No red/stop opportunities; no rule-handling claim. Expert crashed. | [Drive and uninterrupted video](navigation_train_long_drive_20261020.md) |
| Original camera rule tests | CNNfacts15/72passes,22contacts;0/36pairs pass both arms. | [Rule grid](navigation_m3_camera_rules_20261020.md) |
| Corrected label efficiency | Five encoders×four label budgets×three seeds; comma best mean full-label offline score, no safe-driving conclusion. | [Curves and metrics](navigation_m4_repaired_20261020.md) |
| Matched native encoder pilots | All30planned main cases settled;27observed/3unknownfailed. ResNet6arrivals/0contacts but2rulefailures. Originalcomma4arrivals/10contacts. | [Native comparison](navigation_m4_matched_native_20261020.md) |
| Structured stop countermeasure | Both1000update arms rejected; higher recall raises false holds beyond protection. | [Complete tradeoff](navigation_m2_semantic_correction_20261020.md) |
| Observed alternative actions | Brake avoids contact; go/wait collide in both repeats. JEPA/RSSMmiss14/14positive horizons;CVoverconservative. OneTRAINscenario. | [Predictions versus observations](navigation_m5_fixed_plan_diagnosis_20261020.md) |
| Own actuator interface | Approvedversionedinput;160nativeobservations/0mismatches. Probecontacts retained; no planner activated. | [Integration scope](navigation_ego_feedback_integration_20261020.md) |
| Dense-label data engine |3RGB21.97fps versus9camera10.19fps; exact pixel/pose alignment; TRAINlabels never policy inputs. | [Visual label profile](navigation_label_rig_profile_20261020.md) |
| New audited labels |13attempts,11verifiedsources,2quarantined;2200RGB/429dense-label timestamps.9TRAINsources/2predeclared calibration sources. No fit yet. | [Source integrity](navigation_dense_labels_20261020.md) |
| Frozen perception diagnostic | 78 calibration timestamps; raw cross-view object coverage 69.3%, projected 54.3%, fused 53.6%. Vehicle masks only; no pedestrian evidence. | [Coverage, geometry and fixed examples](navigation_dense_perception_20261020.md) |
| Shared RGB observations / depth baselines | All6,124accepted boxes retained,1,269unlocated;429packets independently reproduced. Both box-depth fits rejected; original flat ground remains better where valid. | [Matched depth comparison and limits](navigation_box_depth_20261020.md) |
| New causal-action pilot |10attempts;8verified/2unknown;640RGB/1,280native controls,6exact observable replay comparisons. Front/cross go collide; waiting/braking avoid contact in4s. Rear has0contacts, so missing causal coverage remains. | [All outcomes and native lifecycle limits](navigation_intervention_pilot_20261020.md) |

Current next action: complete the fixed spatial-context TRAIN comparison and its
full-distribution audit. The latest expanded-background head failed; the existing
component result of 55/72 still fails yellow protection. M5 first-event models
have already been fit and rejected; see the later report below.
Preserve all source splits, original M4primary and unrun frozen Town05final suite.

Live command: `.venv/bin/python scripts/watch_navigation_execution.py`.
Exact source hashes, failed attempts, native recordings and milestones remain
in the linked reports, `reports/navigation_execution_20261018.json` and the journal.

- [First-event causal study and all15candidate audit](navigation_first_event_study_20261021.md): both ensembles rejected; four of four calibration contacts missed; graph included.

- [Native bulb-label comparison](navigation_signal_pixels_20261021.md):44/72joint point detections,37,241backgroundpeaks; failed protection gate, all checkpoints retained.

- [Hard-background correction](navigation_signal_hard_pixels_20261021.md):backgroundpeaks94.7%lower,butjoint44→27; rejected, parentretained.

- [RGB context verifier](navigation_signal_context_20261021.md):allcheckpointsrejected,latest12/72joint; TRAIN/VALgapandgeographiccoverageaudited.

- [Strict TRAIN signal diversity and matched fit](navigation_signal_diversity_20261021.md):164/200casesverified,21locations;36unknownretained;newfitrejected,lossselects0.

- [Frozen pretrained/RGB verifier and size diagnosis](navigation_signal_pretrained_20261021.md): both rejected; ResNet 26/72 joint lights despite 97.9% fewer background peaks.

- [Paired tight-crop verifier](navigation_signal_tight_20261021.md): rejected at 28/72 joint; exact RGB control; TRAIN/VAL generalization gap measured.

- [TRAIN-junction regularization](navigation_signal_regularization_20261021.md): all 16 fits converge; TRAIN-selected C=0.0001 yields 30/72 joint lights, still rejected.

- [Nonlinear RGB with TRAIN-selected augmentation](navigation_signal_rgb_boost_20261021.md):41/72joint and95.45%fewer background peaks; three yellow losses still fail coverage.

- [State-aware RGB supervision](navigation_signal_phase_boost_20261021.md):44/72joint and96.86%fewer background detections, but yellow/town protections still fail.

- [Relative-color representation comparison](navigation_signal_opponent_20261021.md): rejected by TRAIN-only selection; raw fold/final predictions reproduce exactly, unselected component VAL untouched.

- [Frozen-proposer fusion protocol](navigation_signal_proposer_fusion_protocol_20261021.md): historical registration; extraction/audit/fits subsequently completed, with rejection documented below.

- [Frozen-proposer feature fusion](navigation_signal_proposer_fusion_20261021.md): rejected by TRAIN-only selection; exact raw control and independent cache/prediction audit retained.
- [TRAIN-only spatial support diagnosis](navigation_signal_train_support_20261021.md):2006/3371positivepoints lack auditedactivebulbs in detailed5×5input; no label/model/validation change.
