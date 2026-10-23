# Navigation experiment runbook

**PAUSED BY USER — 2026-10-23 17:17 IST.** The M0–M6 goal is paused at its actual completed state; it is not achieved. No training/CARLA/queue/guardian process was live at the pause check. Do not launch or resume project runs until the user explicitly asks. Last full signal audit: 1,228/1,248 correct and 3,221 background, rejected. The 1,596-positive cache is verified; its next fit is prepared but unstarted. M2/M3/M5 remain unqualified and M6 final driving is unrun. Documentation and Git checkpoint work are authorized separately. See `reports/navigation_pause_20261023.json`.

**Latest execution — 2026-10-23 17:06 IST:** User-requested conference paper COMPLETE: `reports/conference_20261023/Auto_Thinker_Engine_Conference_Paper.docx` (7 pages, 7 figures, 6 tables, 11 references; only six personal/acknowledgment placeholders bold). Rendered layout/ZIP/placeholder checks pass. Recovered terminal Oct22 evidence: spatial full TRAIN audit REJECTED at 1,228/1,248 joint and 3,221 background; independent recount exact. Additional 1,596 positive cache verified (4,788 feature rows), next three-fold fit prepared but NOT STARTED. Fresh host inventory found no training/CARLA/guardian owner. No new training, job edits or pushes during document work. M2/M3/M5 unqualified; M6 NOT RUN.

**Latest execution — 2026-10-22 22:17 IST:** Three spatial-context fits COMPLETE (508.46 s). Sampled joint 1,144→1,138; backgrounds 201→163: recall tradeoff, no qualification. Full 12,362,313-point TRAIN audit RUNNING at `outputs/navigation_signal_spatial_full_20261022`, 1,200 s / 600 MB cap. Exact compiled CPU raw extraction plus new spatial grids; original sampled/cache/OOF equality checked. Original 1,234 joint/all-town-phase/BG≤3,075.8 gates unchanged. Source bytes pinned. Original guardian and storage budget remain; M2/M3/M5 unqualified, M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-22 22:09 IST:** Spatial RGB cache COMPLETE (146.43 s): 233,110 fixed TRAIN rows ×3 versions ×288 features; 15,417 direct-pixel feature rows exact across 63 fixed images, same whole-junction folds. Three fixed spatial-context fits RUNNING at `outputs/navigation_signal_spatial_fit_20261022` (owner 52141). No component VAL/final fit. Original ≥1,234 joint/all-town-phase/BG≤3,075.8 gates unchanged. Prior expanded-background candidate rejected at 1,232/4,848. Staged CUDA profile passes 236,284-point feature parity/zero decisions changed, median 22.87 ms excluding proposer/control; not deployed. Guardian PID 175563, original storage baseline, 16.62 GB free. Sources pinned. M2/M3/M5 unqualified; M6 NOT RUN.

**Latest execution — 2026-10-22 22:02 IST:** Expanded-background full TRAIN audit COMPLETE/REJECTED: 1,232/1,248 joint (baseline 1,234), 4,848 background (baseline 8,197; required ≤3,075.8). All 12.36M coordinates scored; 48,865 original and 182,016 extra feature rows exact. No component follow-up. Compiled CPU preprocessing passed 236,284-point equality and reduced median feature-plus-head time 973→49 ms; staged only. CUDA parity/profile RUNNING at `outputs/navigation_signal_context_cuda_profile_20261022`, source bytes pinned. After interruption, fresh host inventory found no owners; original storage guardian resumed with unchanged baseline, 17.21 GB free. M2/M3/M5 remain unqualified; M6 NOT RUN.

**Latest execution — 2026-10-22 20:47 IST:** Three expanded-background fits COMPLETE (77.45 s). All 182,016 new TRAIN negatives verified; 12,096 feature rows independently reproduced. Sampled joint detection remains 1,144/1,248 with 201 background detections and small subgroup regressions; no qualification. Full comparison RUNNING at `outputs/navigation_signal_background_full_20261022` (session 18923): all 12,362,313 fixed coordinates, 4,500 s / 600 MB cap, unchanged gates. Source bytes PINNED until closure. Original storage guardian PID 7272; 17.67 GB free. M2/M3/M5 unqualified; M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-22 20:38 IST:** Expandedbackgroundcache COMPLETE:182,016newTRAINnegatives/2,844images×3RGBversions, no oldcoordinateoverlap/unknowns/modelscoremining. Threejoint100wholejunctionfits RUNNING at `outputs/navigation_signal_background_fit_20261022`; independenteligibility/63imagefeaturepreflight, sameweights/hyperparameters/threshold.12focusedtests pass; sourcebytesPINNED. Sampledmetricscannotqualify; laterfull12.36Mpointauditmustprotect1,234joint/alltownphase/BG≤3,075.8. GuardianPID7272/17.688GBfree/originalbudget. Latestcomponent55/72stillFAILSyellow;M2/M3/M5unqualified,M6NOTRUN;goalactive. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-22 20:28 IST:** FullplateauTRAINaudit COMPLETE/REJECTED:1,234/1,248joint but8,197background>3,075.8gate; all12,362,313points/2,844images,48,865sampledrows exact. Independentrecountmatches;5,529FPsnewcoordinates/only201inoldsampledbackground. Report `reports/navigation_signal_plateau_20261022.md`/gallery. No componentfollowup. Next preregister model-independent TRAINnegative sampling fromnewproposaldistribution, strictjunctionfolds; no newfit yet. Allowners terminal/sourcepinsreleased; hostretryafterreviewtimeout confirms onlyguardianPID7272,17.742GBfree,originalbudget. M2/M3/M5unqualified,M6NOTRUN;goalactive.

**Latest execution — 2026-10-22 13:36 IST:** Fullplateau TRAIN audit stillRUNNING at `outputs/navigation_signal_plateau_train_20261022` (owner14275/PID40722). Partial1114images already3352background>wholeTRAIN3,075.8limit: cannotqualify; finishregisteredfullcross-townevidence, no componentfollowup. Sourcepinsremain. Stagedpoint-rayinterface3tests/GPUtreeexport4096TRAINrowsparity6.66e-16/0decisionchanges complete, NOTintegrated; GPUkernel.815msvsCPU3.519ms/batch256, notfullruntime. See `reports/navigation_signal_runtime_staging_20261022.md`. GuardianPID7272/originalbudget; M2/M3/M5unqualified,M6NOTRUN;goalactive. Live: `.venv/bin/python scripts/watch_navigation.py --out outputs/navigation_signal_plateau_train_20261022`.

**Latest execution — 2026-10-22 13:19 IST:** Plateau RGBinventory COMPLETE/PASS:12,362,313points, candidatecoverage1,209→1,246/1,248,37newlycoveredmisses/43recognizedoracle-membershipcases; noaccuracyclaim. Full classifier TRAIN audit RUNNING at `outputs/navigation_signal_plateau_train_20261022`, existingOOFheads/no fits,3600s/600MBcap. Requirejoint>1,189/everytownphaseprotected/BG≤3,075.8beforefurthercomponentcheck. Elevenfocusedtests pass; sourcebytesPINNED. GuardianPID7272,18.26GBfree/originalbudget. Latestcomponent55/72stillFAILSyellow12vs15. M2/M3/M5unqualified,M6NOTRUN;goalactive. Live: `.venv/bin/python scripts/watch_navigation.py --out outputs/navigation_signal_plateau_train_20261022`.

**Latest execution — 2026-10-22 13:15 IST:** All-pixelRGB inventory FAILED at15Mpointcap after2164completeimages/45.17s; no classifierpredictions, no finalcoordinateartifact, failurekept. Distinctregisteredplateau-consolidated inventory RUNNING at `outputs/navigation_signal_plateau_inventory_20261022`:onecentroid-nearestpixelperchannelmaximumregion, everyregion/oldpointretained, same5×5/16contrast. No fit/inference/componentVAL. Eightfocusedtests pass; sourcesPINNED. GuardianPID7272/originalbudgetunchanged. Latestcomponent55/72stillFAILSyellow12vs15. M2/M3/M5unqualified,M6NOTRUN;goalactive.

**Latest execution — 2026-10-22 13:12 IST:** RGB refinement component COMPLETE/REJECTEDyellow:55/72joint,127background,red30/green13/yellow12(parentyellow15),towns7/17/12/19. Every38,543oldfeature/probabilityrowexact; independentrecountmatches. Report `reports/navigation_signal_refinement_20261022.md`. RemainingTRAIN59misses:39nocandidate/20filtered;49recognizableoraclepointsabsent(neverruntimeinputs). IndependentRGBglobalproposal inventory RUNNING at `outputs/navigation_signal_global_inventory_20261022`:fixed5×5channelmax/localcontrast16,no classifierpredictions/fit/VAL. Sevenfocusedtests pass, sourcesPINNED. OriginalguardianPID7272, budgetunchanged. M2/M3/M5unqualified,M6NOTRUN;goalactive.

**Latest execution — 2026-10-22 13:06 IST:** RGB refinement TRAIN audit COMPLETE/PASS: joint1,144→1,189/1,248, background724→2,668(still91.33%belowraw30,758), everytown/phaseprotected. Alloriginal715,825points retained;1,054,818added. Exact48,865sampledrows. Incrementalcostmedian.091s/image, notfullruntime. Conditionalcomponentcheck RUNNING at `outputs/navigation_signal_refinement_component_20261022`; existinghead, nofit, unchanged72labels/gates. SourcesPINNED. OriginalguardianPID7272/18.286GBfree. M2/M3/M5unqualified,M6NOTRUN;goalactive.

**Latest execution — 2026-10-22 11:49 IST:** TRAIN appearance diagnosis COMPLETE:104misses=58no proposal+46filtered; existing OOFheadrecognizes94at auditedactivepixels (oracle diagnostic only, prohibited as runtimeinputs). RGB-only localrefinement audit RUNNING at `outputs/navigation_signal_refinement_train_20261022`: keep everyoriginalpoint/add5×5R/G/Bmaxima, sameheads/.5acceptance, nofit/componentVAL. Eightfocusedtests pass; allnavigation sourcebytesPINNED. SameoriginalbudgetguardianPID7272/18.35GBfree. M2/M3/M5unqualified,M6NOTRUN;goalactive. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-22 11:38 IST:** Supported-supervision study COMPLETE, component REJECTED. TRAIN full audit improves joint1,141→1,144/1,248 and background30,758→724; all48,865sampledrows exact. Original component evaluation43/72joint/40background (99.89%fewer) still FAILS total44→43, yellow15→10, Town10HD10→8. Independent component recount exact; tenfocusedtests pass. Report `reports/navigation_signal_supported_20261022.md`/graph. Allowners terminal, sourcepinsreleased/archivesretained; onlyoriginalguardianPID7272live,18.35GBfree. NextTRAIN-onlyphase-appearance/generalizationaudit before another preregisteredfit; no threshold/benchmarkchange ornativepromotion. M2/M3/M5unqualified,M6NOTRUN;goalactive. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-22 11:35 IST:** Full TRAIN audit COMPLETE/PASSES: all715,825proposals/2,844images; correcthousing-phase1,141→1,144/1,248, background30,758→724(97.65%fewer), alltowns/phasesprotected. All48,865sampledfeatures/probabilities exact, independentrecountmatches. One registered finalfit/componentevaluation RUNNING at `outputs/navigation_signal_supported_final_20261022`; unchanged144images/72labels/gates, no deployment. Tenfocusedtests pass; sourcebytesPINNEDuntilterminal. OriginalbudgetguardianPID7272,18.36GBfree. M2/M3/M5unqualified,M6NOTRUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-22 11:30 IST:** Supported-supervision TRAIN study COMPLETE/PASSES internal check: joint1,141→1,144/1,248, sampled background1,990→201, every town/phase non-regressing. Original point loss WORSENS.360→1.264; no component VAL/deployment. Full TRAIN audit now RUNNING at `outputs/navigation_signal_supported_full_20261022`: all715,825savedproposals/2,844images, existingfoldheads, no newfits. Require exact48,865sampled-row parity and independentrecount. Sourcebytes PINNED; sameoriginal14GBnet/10GBminimumguardianPID7272. M2/M3/M5unqualified,M6NOTRUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-22 11:27 IST:** TRAIN-only housing/context audit COMPLETE: 2,006/3,371 positive proposals lack an active pixel within detailed 5×5 RGB; 1,351 of those have alternate-phase differences within measured red-repeat variation. No overlapping boxes explain it; fixed examples show housing edges/support structures. Three supported-supervision CPU folds now RUNNING at `outputs/navigation_signal_supported_20261022`, original unsupported positives still scored. No component VAL, all-TRAIN fit or deployment. Six focused tests pass; navigation sources PINNED until study terminal. Guardian resumed SAME original baseline after fresh host check found no owners; PID7272, 18.01GB free, 14GB net cap/10GB minimum unchanged. M2/M3/M5 unqualified, M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 17:34 IST:** Proposer feature cache/audit/fusion study COMPLETE. Fusion loses TRAIN-onlyselection(.365732versusraw.360094); rawpredictionsreproduceexactly, unselectedfusedhead neverevaluatedoncomponentVAL. Selected44/72joint/1171backgroundstillFAILSyellow/Town10protection. Report `reports/navigation_signal_proposer_fusion_20261021.md`. TRAIN-onlydiagnosis finds2006/3371positivepoints lackan auditedactivebulbpixelwithin detailed5×5RGB; correctjoint1253/1365withsupportversus360/2006without. Conservative-mask/locationconfoundsdisclosed; no relabeling. See `reports/navigation_signal_train_support_20261021.md`. Next audit supervision/context onTRAINbeforenewfit. Allowners terminal; sourcepins released, archives retained; freshhostcheckonlyguardianPID2309550. Original14GBnet/10GBminimum baseline,17.98GBfree. M2/M3/M5unqualified,M6NOTRUN;goalactive.

**Latest execution — 2026-10-21 17:27 IST:** Frozen-proposer RGB fusion FIT RUNNING at `outputs/navigation_signal_proposer_fusion_20261021`. Cache v2complete107.03s/all8676sameforwardchecks exact; independentauditcomplete14.20s/67fixedforwards exact across21TRAINlocations and4VALtowns. All originalRGB/features/points/modelweights verified. Oldsavedscore difference max.013758,zero pointclasschanges, disclosed. Originalzero-featureJSONtuplepreflightfailure retained; same originaldeadline. Cache/audit owners exited0; classifiernowpins itsregistered sources. Raw87 versusfused123features, same400trees andTRAINlocationselection; onlyselectedhead reachescomponentVAL. Ninefocusedtests pass; no nativejob/deployment. Originalstoragebaseline14GBnet/10GBminimum unchanged. M2/M3/M5unqualified,M6NOTRUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 17:22 IST:** Frozen-proposer cache RUNNING at `outputs/navigation_signal_proposer_features_20261021`. Host review recovered: actual inventory confirms only originalguardian before launch, GPU10,570MiBfree/disk18.03GBfree. Registered full-image32features+4probabilities, originalpoints/gamma data, same frozencheckpoint; no classifier fit yet. ALL src/navigation source bytes are PINNED until cache closure; do not edit. Ninefocusedtests previously passed. Independent cache audit required before fits. Original storagebaseline14GBnet/10GBminimum unchanged. Previous reviewfailure preserved, nowresolved. M2/M3/M5unqualified,M6NOTRUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 17:21 IST:** Frozen-proposer feature fusion PREPARED, NOT STARTED. Extractor/runner/protocol ready; nine focused CPU tests pass, including exact original full-image probability parity and frozen weights. No GPU cache or newfit executed. Protocol `reports/navigation_signal_proposer_fusion_protocol_20261021.md` discloses upstream TRAIN exposure and keeps all original gates. Final host process/storage check was NOT executed because automatic approval review hit its usage limit; do not bypass or infer host ownership from statusfiles. Last successful host inventory17:03saw guardianPID2309550; priorstudy handle exited0, current guardianheartbeat fresh. Recheck hostowners/GPUreadiness after reviewrecovers before launch. Same immutable14GBnet/10GBminimum budget; no artifacts deleted. Latest completed model44/72joint/1,171background stillFAILSyellow/Town10. M2/M3/M5unqualified,M6NOTRUN; goalactive.

**Latest execution — 2026-10-21 17:09 IST:** State-aware and opponent RGB studies COMPLETE, neither qualified. State-aware44/72joint/1,171background loses yellow15→10 and Town10HD10→9, so aggregate equality does NOT pass. Opponent features lose TRAIN-location selection(.368144versusraw.360094); unselected arm never evaluated on componentVAL. All raw-control fold/final predictions reproduce exactly. Reports `navigation_signal_phase_boost_20261021.md` and `navigation_signal_opponent_20261021.md`. No active fit/native owner; sourcepins released afterclosure, archives retained. Only originalguardianPID2309550 remains, same14GBnet/10GBminimum baseline. Next register frozen-proposer visual feature fusion; no newfit started. M2/M3/M5 unqualified, M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 17:05 IST:** Opponent-color RGB comparison RUNNING at `outputs/navigation_signal_opponent_20261021`: same joint100classifier,87raw versus312RGB-derived features, exact-control TRAIN prediction parity. Seven bounded CPU fits, only TRAIN-selected final head reaches component VAL. Ten focused tests pass; source bytes PINNED. Previous state-aware study COMPLETE but FAILED:44/72joint,1,171background peaks; yellow15→10 and Town10HD10→9 violate protection. Report `reports/navigation_signal_phase_boost_20261021.md`. No native job/deployment. Same original storage guardianPID2309550,14GBnet/10GBminimum. M2/M3/M5 unqualified, M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 16:58 IST:** State-aware RGB study RUNNING at `outputs/navigation_signal_phase_boost_20261021`: joint100 four-class versus binary400, each400trees, same augmented TRAIN cache and whole-junction folds. Seven bounded CPU fits; only TRAIN-selected head reaches unchanged component VAL/gates. Ten focused tests pass. Source bytes PINNED until terminal; no native job. Prior gamma-binary result41/72joint/1,696background stillFAILED because three yellow lights are filtered; no threshold change or deployment. Original storage guardianPID2309550 /14GBnet/10GBminimum unchanged. M2/M3/M5 unqualified, M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 16:54 IST:** Nonlinear RGB study COMPLETE, still REJECTED. Seven fixed-budget CPU fits select gamma augmentation on TRAIN-junction folds; final47/72localized,41joint,1,696background peaks (95.45%below parent). Three parent-correct Town10HD yellow lights are filtered out; original-order diagnostic does not recover them. Same required44joint and per-town/state protection; no promotion/deployment. Independent OOF/selection/component recount passes; report `reports/navigation_signal_rgb_boost_20261021.md`. All study processes exited; source pins released, exact archives retained. Only original storage guardianPID2309550 remains, resumed with unchanged baseline14GBnet/10GBminimum. Empty preflight failure and original deadline preserved. Next: prospective state-aware RGB supervision versus matched-capacity binary control using TRAIN-only folds. M2/M3/M5 unqualified; M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 16:47 IST:** Nonlinear RGB / gamma study RUNNING at `outputs/navigation_signal_rgb_boost_v2_20261021`: seven bounded CPU fits, whole-TRAIN-junction selection, same 144 VAL images /72 labels and protection gates. Registered source bytes PINNED through closure. Original output retains an empty failed preflight (stale storage heartbeat; zero images/fits). Host check after a five-hour execution gap found no owner; guardian resumed the SAME immutable 27,792,580,608-byte baseline /14 GB net cap /10 GB minimum. New invocation retains the original 30-minute deadline; no scientific case or model fit repeated. Seven focused tests pass. No native job or deployment; M2/M3/M5 unqualified, M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 11:36 IST:** All current fits SETTLED; only original storage guardian remains active. TRAIN-junction regularization completes 16 converged fits and selects C=0.0001 before component VAL. Final 30/72 joint lights / 1,919 background peaks still FAILS parent coverage of 44. Independent fold-selection and component recount pass. See `reports/navigation_signal_regularization_20261021.md` / graph. Paired tight crops also failed at 28/72; prior large-crop verifier failed at 26/72. No promotion or driving rollout. Next: representative error inspection, then prospective TRAIN-only augmentation / feature adaptation using whole-location folds. Sources released only after terminal; archives retained. Same storage baseline, 14 GB net cap / 10 GB minimum, 18.07 GB free. Ego feedback remains applied/native-verified; M2/M3/M5 unqualified, M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 11:32 IST:** TRAIN-junction regularization RUNNING at `outputs/navigation_signal_regularized_20261021`: 15 internal-fold fits across 21 TRAIN locations, then one TRAIN-selected final fit and fixed component validation. Cached features reused; no GPU/native job. Five focused selection/boundary tests pass. Registered source bytes are PINNED until terminal. The paired tight-crop study FAILED at 28/72 joint lights / 2,250 background peaks, with exact RGB-control parity. See `reports/navigation_signal_tight_20261021.md`. Prior large-crop model has 99.7% TRAIN / 72.9% VAL balanced accuracy; no blind extension. Same storage guardian baseline, 14 GB net cap / 10 GB minimum, about 18.1 GB free. M2/M3/M5 unqualified; M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 11:25 IST:** Paired tight-crop verifier RUNNING at `outputs/navigation_signal_tight_20261021`. Same TRAIN points, original 144 VAL images / 72 labels, fixed gates; only frozen ResNet crop scales change to 8/16 native pixels. RGB repeats as an exact-parity control. Eight focused tests pass. Registered sources are PINNED until terminal. Prior pretrained verifier FAILED: ResNet 26/72 joint lights, 789 background peaks; RGB 38/72 and 9,176. See `reports/navigation_signal_pretrained_20261021.md` and graph. Only this fit and original storage guardian are active; no native driving. Same 14 GB net cap / 10 GB minimum. M2/M3/M5 unqualified; M6 NOT RUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

**Latest execution — 2026-10-21 10:55 IST:** Signal-diversity collection and fit SETTLED. Independent audit accepts164/200cases (1968RGB,21TRAINjunctions);36Town01clear-noon cases remain unknown after weather-state failure, never retried. Original collector staysFAILED; separate verifiedmanifest retained. Newsame-recipe1000updatefit FAILS: lossselects0; latest61/72localized,43joint,88,379backgroundpeaks versusold56/44/37,241. No modelpromotion. See reports/navigation_signal_diversity_20261021.md/graph. Priorcontextverifieralsofailed12/72joint. Allnative/fit/auditowners exited; onlyoriginal14GBnet/10GBmin capacityguardian remains. Sourcepins released afterterminal; exactarchives retained. Next: prospectivepretrainedRGBforegroundverification/representationdiagnosis, preserveallreservedlocations and original144VAL/72labels. M2/M3/M5unqualified,M6NOTRUN;M4primary unchanged. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.


**Latest execution — 2026-10-21 10:46 IST:** Strict TRAIN signal diversity capture RUNNING at outputs/navigation_signal_diversity_20261021:200planned cases/21junctions/twoillumination settings. FirstTown01clear-noon worker failed its weather-state check before capture (36unknown, no retry); remaining frozen tasks continue. Independent audit waits for exactcollectorPID2206711 at outputs/navigation_signal_diversity_audit_20261021. All src/navigation and registered capture dependencies remain PINNED through terminal audit; do not edit them. Original14GBnet/10GBmin guardian active, about18.8GBfree. RGBcontext verifier finished and FAILED:latest12/72joint,TRAIN/VALbalancedaccuracy82.5%/40.2%, no selectedverifier. See reports/navigation_signal_context_20261021.md. No driving promotion;M2/M3/M5unqualified,M6NOTRUN. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.


**Latest execution — 2026-10-21 00:57 IST:** Two new camera-signal studies COMPLETE, no promotion. Audited paired phase labels:73eligibleTRAINposes/sevenlocations,65excludedoldTRAINposes; original144VAL/72labels unchanged. Native bulb CNN gives56/72point-localized,44joint but37,241backgroundpeaks: FAIL. Hard-background correction reducespeaks94.7%butjoint44→27: FAIL; selectunchangedparent0. See reports/navigation_signal_pixels_20261021.md and navigation_signal_hard_pixels_20261021.md/graphs. Seventeen focusedtests pass. Alltrain/nativeworkers exited; only originalstorageguardian live. Next: widerRGBhousingcontext/TRAIN-onlycandidate verification, freshprospectiveprotocol beforefit. M2/M3/M5unqualified, M6NOTRUN; originalM4primary/benchmarks unchanged. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.


**Latest execution — 2026-10-21 00:44 IST:** First-event study COMPLETE and both ensembles REJECTED. Six bounded fits select100/0/0; JEPA/RSSM Brier.17706/.17728 lose to action-only.16328 and miss all4calibration contacts. Frozen all15candidate audit confirms rejection; unexecuted outcomes remain unknown. See reports/navigation_first_event_study_20261021.md and graph. Combined eligible data68/73attempts,50TRAIN/18calibration; original failures preserved. No native planner or M6 driving. Only original capacity guardian live (14GB net/10GB minimum, unchanged baseline). Next: audit paired active-bulb supervision and current geography exclusions before new camera perception fits. All prior gate failures and original M4primary remain unchanged. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.


**Latest execution — 2026-10-21 00:10 IST:** Approved ego-only actuator feedback v2 is already applied and native-verified (160 observations, zero acknowledgment differences); no planner is active. The original ten-attempt intervention pilot settled with eight independently verified factual TRAIN recordings and two unknown failures, never retried. Six observable world-prefix/repeat comparisons are exact; three observed contact starts, no positive rear or red outcomes. Original source owners/audit remain FAILED. See `reports/navigation_intervention_pilot_20261020.md` and separate verified manifest.

The 63-case frozen expansion has now SETTLED in 759.2 seconds: 60 child captures (42 TRAIN / 18 calibration), three failed TRAIN attempts retained. Engine exits: 62 signal-11 and one exit143; no engine-stability claim. Independent closed-file audit RUNS at `outputs/navigation_intervention_expansion_audit_20261020`. Source remains PINNED through this audit; do not edit navigation runtime or audit dependencies. New Python-child-per-owned-engine containment keeps the parent alive; it does not cure the engine. No retries, automatic fit, driving promotion or final Town05 run. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

A retained client core identifies LightManager's blocking native light-update callback as one SIGABRT path, not the cause of every engine SIGSEGV. Two containment instruments exited child0 but engine-11; earlier lease probe FAILED. The original storage guardian resumed the SAME approved baseline after an execution gap; 14GB net cap / 10GB minimum unchanged. Around 18.9GB free and 8.9GB net consumed. M2 reliability/M3 rules/M5 planning remain unqualified; M6 NOT RUN. Original M4 primary and every benchmark denominator remain unchanged.


**Latest execution — 2026-10-20 16:07 IST:** The bounded ten-attempt
TRAIN action pilot is RUNNING at `outputs/navigation_intervention_pilot_20261020`:
static lead, moving crossing and rear approaches; fixed go/brake/wait actions,
plus one go repeat. 900s/600MB cap, independent native/pixel/plan/geography audit
before any dynamics fit. Scripted actors are not reactive traffic. Source bytes
are PINNED: all `src/navigation/*.py`, current collector/audit/dependency scripts,
config and archived geometry. Do not edit them or start a duplicate owner.
Stop only with this queue's STOP. Live: `.venv/bin/python scripts/watch_navigation_execution.py`.

Completed perception diagnosis: 78 calibration timestamps, cross-view object
coverage raw69.3% / projected54.3% / fused53.6%; vehicle masks only. New optional
RGB observation pack retains all6,124boxes, including1,269unlocated. All429packets
independently reproduce; original/new detector parity passes. No driver has adopted
it. Two cheap box-depth replacements both REJECTED: common-pair encoded-depth
MAE flat1.53m / boosting1.77m / ridge3.15m. No blind continuation or deployment.
Native depth-convention probe failed before capture (0/6observed, sixunknown),
retained/no retry. Native convention is unconfirmed. New future-only cleanup helper
skips RPCs to an exited owned engine; tests pass. See
`reports/navigation_dense_perception_20261020.md`,
`reports/navigation_box_depth_20261020.md` and the results index.

Geometry v1draft retained. Before any native outcome, v2corrects an unreachable
crossing start and keeps rear actors on guarded routes; all8eligible verified
TRAINsources plus original2calibration sites considered.24usable /6unavailable
of30geometric contexts retained; no1mclearance or location-role relaxation.
The old route6branch diagnostic stays outside new TRAIN. Current pilot uses only
r35_lead/r134_crossing/r35_rear,10original attempts, no retries or auto-fitting.

Approved actuator-feedback interface remains applied/nativeverified; M4primary
and original comparisons unchanged. Familiar ResNet1.376kmarrival has0contacts
but0red/stop opportunities, so no rule claim. Original M3fails15/72,0/36pairs;
M5oldmodels rejected, M2safety gates open, M6NOTRUN. All failed evidence preserved.
Dense labels remain11verified sources/2quarantined:351TRAIN/78calibration timestamps;
never fit the22quarantined labels. Same original14GBnet/10GBminimum storage guardian.
25targeted collector/boundary/feedback tests pass plus7new perception/cleanup tests;
full historical485test suite was not rerun for these isolated additions.

**Latest execution —2026-10-20 15:21:** All13dense-label sourceattempts SETTLED.
Originalroute78zero-labelspawnfailure retained;independentauditquarantinesWHOLE
source101for0.501mclearance versusrequired1m. Originalcollector/auditremainFAILED;
separate verifiedmanifest includes11sources(9TRAIN/2calibration),2200RGB/429labels,
allpixels/nativeacks/actualgeographychecked. Source451labels totalincludes22
quarantinedsource101labels;do NOTfitthose. CalibrationTown03/1221andTown04/916
fixedbeforecollection. All5TRAINtownscovered. No newfitstarted. See
reports/navigation_dense_labels_20261020.md and navigation_results_index_20261020.md.
Allnative/collection/auditownerprocessesareterminal;onlystorageguardianremains.
Newruntime/labelsourcegenerationisarchived;do notresumererunoriginalattempts.
OriginalM4matrixcomplete(noqualifieddriver),approvedactuatorfeedbackapplied/160
nativeacks0diff. ResNet1.376kmcameraTRAINdrivearrives0contacts/0interventions,
5speedinfractions. Its native meter has0red/0stopOPPORTUNITIES,soNOlearned-rule
claimfromthatdrive. M3CNNfails15/72,0/36botharms;M5modelsunqualified;M6notrun.
NextinspectexistingFPNagainstnewpixel/depthcalibration,thenregistersharedperception
anddiversefirst-eventactioncoveragebeforefitting. Livewatcher accuratelymarks
11accepted/2quarantined;full485CPUtests/2skipplusauditor1testpass. Approved
14GBnetcap/10GBminimumfreeguardianstayssamebaseline,~5.98GBused/21.8GBfree.


**Latest execution —2026-10-20 15:10:** Matched M4native matrix COMPLETE:
5encoders/30planned,27native/3unknownfailed. ResNet6/6arrivals/0contacts/1red1stop;
originalcomma4/6arrivals/10contacts/2red1stop. No newprimary or safety promotion.
Approvedego-feedback interface APPLIED and nativeverified160observations/0mismatches;
its2probecontacts remain counted. No trajectoryplanner activated.
NewfrozenResNetcamera1.376kmstrictTRAINlongdrive ARRIVES0contacts/0red0stop/
0interventions/5minimum-speedinfractions.3990actualposesTRAINonly, IPCp9525.1ms;
full199.5svideo retained at reports/navigation_train_long_drive_20261020.md.
Originalexpertsegfaultbeforefirsttracerow retained; no validexpertcomparison.
Nine-cameralabelprofilecomplete:3RGB21.97fps versusRGB+instance/depth10.19fps,
allrawpixel/posealignmentverified,0contacts,142.61MB. TRAINonlylabeladapter
passes485fullCPUtests/2skip plus independentauditor tampertest1pass. Current
13source/5townlabelscollect at outputs/navigation_dense_label_continuation_20261020
(originalroot navigation_dense_label_collection_20261020 retainedfailed4/13).
Route78actualspawnTown02/242reserved guardstops beforelabels; neverretry it.
Only9originalunstartedcases continue withsameplan/source/roles/shared3GBcapand
conservativeoriginal1800sdeadline. Current9/13attemptssettled; oneisfailed,
progress≠success. CPUauditqueues at outputs/navigation_dense_label_audit_20261020;
first3partialaudits verify600RGB/117denselabels/0ackdifferences/actualgeography.
ALLcurrent src/navigation, run_navigation.py, navigation_label_rig.py, collector
and configs are SOURCE-PINNED through collection/audit. Do NOT edit thosebytes.
Newlabels stay outside policy/teacherinputs; nofitstarted. Guardian original14GB/
10GBbudget active;5.17GBused/22.6GBfree at15:09. M2rule/300encounter,M5andM6
remain open. Read ledger/journal/live watcher before any next native launch.


**Latest execution —2026-10-20 14:11:** Protected semantic correction COMPLETE: both
1000update arms select0; facts750raises stop recall55.0→59.46% but falseholds
13.92→16.32%, violates+1point protection. No new driver promoted.
Fixed-action pilot6/6closed/audited,480RGB/960acks/0differences, all observable
replay tests pass,4contacts (go1.30s/wait-go3.80s eachrepeat; brake0).
M5frozen CPUdiagnostic: JEPA/RSSMmiss14/14positive horizons;CV0misses but
34/34negative horizons above.5. All UNQUALIFIED, one TRAINscenario only.
See reports/navigation_m5_fixed_plan_diagnosis_20261020.md and action association
audit. New20Hzfirst-event prototype is UNTRAINED/staged;3structural tests pass.
Matched-native comparison12/30settled: CNN0/6arrivals/1contact/2unknownfailures;
ResNet6/6arrivals/0contacts/2rulefailures; DINOcurrently running. Originalcomma
primary remains unchanged. Source/config stay pinned through comparisons AND
queued feedback activation; do NOT edit. All source hashes still match.
New strictly TRAIN1.376km Town04route geometry verified; onejunction134,
43.29m clearance from reserved geometry. After feedback audit, queue expert and
frozen ResNetengineering long-drive pair at outputs/navigation_train_integration_pair_20261020,
seed62071,900s/arm. Chosenposthoc/familiarTRAIN, not a new benchmark, primary
reselection or proof of urban rules. No drive there yet; old Town03results retained.
See ledger/journal/live watcher for current progress. M6notqualified.


**Latest execution —20 October2026 13:31:** Original CNN-facts M3evaluation COMPLETE,
15/72passes,22unfilteredcontacts across9trials,15red/5stop;0/36location–seed
pairs pass both arms. All72actuation recordings verify,0differences/0missingacks.
See reports/navigation_m3_camera_rules_20261020.md/json/png. No safety promotion.
Protected semantic correction now verifies original RGB before its paired fits.
Then fixed-action6rollouts → remaining matched M4heads/no-color diagnostic.
All current src/navigation/config bytes remain pinned through matched-native
completion; do NOT edit. User approved ego-only actuator feedback;36isolated
interface tests pass plus1activation-guard test. New guarded source activation
queues at outputs/navigation_ego_feedback_activation_20261020 AFTER all four
pinned roots/processes terminate, then one TRAIN160frame/240s native-feedback
probe. It introduces no trajectory planner or qualified driving model. Approved
14GBadditional/10GBreserve storage guardian remains active. M6notqualified.


**Latest execution —20 October2026 11:50:**67corrected fits,4fact/plain fits and
6factual dynamics members complete. Exact remaining-candidate pilot queue:
outputs/navigation_corrected_pilot_continuation_20261020. Stop it with
`touch outputs/navigation_corrected_pilot_continuation_20261020/STOP`.
Original failed first-case evidence stays retained. M2/M3/M5/final safety gates
remain open; no automatic checkpoint promotion. Approved14GB additional net
usage/10GB free reserve has a live guardian. Current25GB source-pinned runtime
changes only after queued clock-repair/compact activation v3. The M6 showcase
subset is fixed before outcomes; every native trace/recorder is retained.
Use `.venv/bin/python scripts/watch_navigation_execution.py` for the live bars.

**Previous execution —20 October2026:** Actual-geography-corrected JEPA continuation
completed10000updates/1768.34s;67fresh matched downstream fits now run at
`outputs/navigation_repaired_study_v4_20261020`. The old66fit study is historical
and split-contaminated. All original13validation episodes/2822timestamps stay
unchanged; corrected55TRAIN episodes/15712timestamps. Four prospective paired
cold-start/fact CNN/ResNet candidates, causal features, GPUFPNtiming, repaired
memory references and six factual dynamics members are queued serially; check
host processes/ledger before any launch.480branch RGB/960actualcontrols audited,
8contactstarts; frozen-light elapsed replay tolerance fails, so no calibration
qualification.24expert rule references remain expert-only; no student M3 or
Town05 final driving. M2safety/M3/M5/M6 remain open. Live:
`.venv/bin/python scripts/watch_navigation_execution.py`. Stop the active study
with `touch outputs/navigation_repaired_study_v4_20261020/STOP`.

**Collision accounting correction —2026-10-19:** The pinned native criterion ignores
near-stationary ego contacts. Official recorder queries verify unfiltered expert
293/300 rule-safe and56/300 strict (6 contact starts across3missions), versus old
native294/57. Long camera drive has13 contact starts versus5 native events; short
pilots remain0. Preserve native scores separately. See
[collision correction](navigation_collision_correction_20261019.md).


This branch adds a separate camera-navigation experiment. The existing Dreamer
results, reward contract, job scripts and best90k checkpoint remain unchanged.
See [the approved plan](navigation_plan.md) and [configuration](../configs/navigation.yaml).

Latest2026-10-19: [camera ablation and long drive](navigation_visual_decision_20261019.md)
completes6/6 pilot routes (4 rule-compliant), then1233.361m integration route with
13 unfiltered recorder contact starts/2 red/1 stop violations. Unsafe, not promoted.
The prior speed-conditioned spatial model stays stationary0/6 despite96.79%
offline recall; keep every negative result and do not expand that prototype.
Actual speed remains in physical tracking; neural speed omission is explicit.

[Historical shared encoder study](navigation_m4_visual_protocol_20261019.md)
completed but has actual-geography overlap; corrected study follows the top banner. Five encoders, nested
733/1257/4069/16639 labels,three seeds,2000 identical-draw updates and six ablations.
No inherited full-label driver/perception head. TRAIN-only JEPA continuation and
no-update feasibility profiles included. One cache at a time,25GB reserve.
Do not edit the source-frozen training dependencies while running or duplicate it.

```bash
.venv/bin/python scripts/watch_navigation_execution.py
# Ctrl-C only stops watching. To stop this study through its owners:
touch outputs/navigation_repaired_study_v4_20261020/STOP
```

The user authorizes needed runs through M6; no repeat permission for routine stages.
[New held-out expert reference](navigation_heldout_expert_20261019.md) is complete:
293/300 recorder-corrected rule-safe, two missing setups retained. The old engineering long
route traverses heldout_dev Town03/1135 and103: disclose exposure, retain all300
cases and avoid tuning those sites. TRAIN/pretraining excludes them; Town05 student
outcomes remain unused. Do not claim fully untouched17-location geography.
Town05_final_v2 remains frozen and final driving still gated by M2–M5 evidence.
Raw-attempt collision/setup/opportunity auditing is implemented; actual final driving remains pending;
never fabricate qualification or drop failed cases. No candidate is qualified.

Previous continuation: fixed causal TRAIN replay, below, remains reproducible.

Latest: [fixed causal TRAIN replay](navigation_m2_signal_temporal_20261016.md).
The selected static bundle has an RGB-only inference wrapper and a causal
per-camera tracker, both still offline components. Do not pass their outputs or
checkpoint to the driving worker as a new camera policy. Applicability remains
unresolved; only GT scoring associates the applicable actor. No new driving gain.

The completed replay is `outputs/navigation_m2_signal_temporal_20261016`.
Do not restart it. Rebuild its report without GPU or model inference:

```bash
.venv/bin/python scripts/summarize_navigation_signal_temporal.py
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_temporal_20261016 --once
```

Its source preflight is read-only and can be checked with:

```bash
.venv/bin/python scripts/audit_navigation_signal_temporal.py \
  --out outputs/unused_preflight_target --preflight
```

It excludes old111/133 at newly reserved component-validation Town03/655;
preserves the original six correction episodes; validates four safe TRAIN routes
and selects the two complete signal-bearing episodes before model inference.
No complete moving approach/onset/wait/release exists in these two recordings:
101 ends37.2m before its API light reference,123 starts4.6m away already waiting.
Next prepare full-approach TRAIN collection with coverage/censoring checks and
verified visible/applicable labels; do not train longer on missing supervision.

## Install and inspect

From the project root, using the existing Python3.10 environment:

```bash
.venv/bin/python scripts/setup_navigation.py --install
.venv/bin/python scripts/run_navigation.py doctor
```

Source and small additional Python packages go under `outputs/navigation_vendor`.
The source revision and package versions are pinned in
`configs/navigation_vendor.json`; the installer does not change `.venv`.
The tracked patch removes PDM-Lite's forced-throttle escape, rejects invalid
actor geometry instead of deleting actors, resolves speed-limit paths relative
to source, handles `_Opt` map names, and fixes a setup-failure reporting call.
This modified privileged expert is not claimed to reproduce published scores.

The optional harness tests require this installation; the core navigation tests
do not. Importing CARLA's Python package does not require a running simulator.

## Short profile and full expert route

The checked-in M0 route is a development mission on Town03_Opt. It is not part
of the final Town05 benchmark. Its original OpenDRIVE path is 1228.73m; the
harness interpolation measured 1233.36m. The route adapter rejects length drift
above 2% or 10m, whichever is larger. The route manifest hashes the XML and map.

```bash
.venv/bin/python scripts/run_navigation.py expert \
  --out outputs/navigation_profile_new --max-steps 200 --max-wall 300

.venv/bin/python scripts/run_navigation.py expert \
  --out outputs/navigation_expert_new --max-wall 900
```

Use a new output directory for every attempt. For a sustained run, prefix the
command with `systemd-inhibit --what=sleep:idle:handle-lid-switch --why=navigation`.
The runtime refuses competing navigation runs and active historical trainers.
It owns one server on port 2100 with Traffic Manager 8100 and uses SIGTERM only
for owned processes. A separate evaluator child contains native client crashes.
The existing CARLA TM cache workaround must already be applied.

The installed CARLA defaults to Town03_Opt despite the command-line map argument.
The adapter loads a different requested map **before** creating Traffic Manager;
all five training towns have passed bounded camera probes through this path.
Reloading with an active Traffic Manager crashed during initial integration.
A map mismatch is a failure, never silently accepted as the requested town.

Watch without changing the run:

```bash
python3 scripts/watch_navigation.py --out outputs/navigation_expert_new
python3 scripts/watch_navigation.py --out outputs/navigation_expert_new --once
```

Stop cleanly:

```bash
touch outputs/navigation_expert_new/STOP
```

Ctrl-C in the watcher only stops watching. The STOP file is checked during
startup and scenario ticks. The owner also polls it; after a 60s grace period it
may terminate its hung evaluator child with SIGTERM. No force-kill fallback is
used. A cleanup failure is preserved in `cleanup_failed.json`.

## Evidence per attempt

- `manifest.json`, `source_hashes.json`, `source/`: config, seed, route and code provenance.
- `leaderboard.json`: original upstream criteria and Driving Score records.
- `summary.json`: all collision events, completion and stricter safe/compliant completion.
- `progress.json`: current measured state; the bar uses evaluator route progress.
- `episodes/*/trace.jsonl`: controls, ego pose/speed, privileged expert diagnostics.
- `episodes/*/world_trace.jsonl`: audit-only actors, velocities and traffic-light phases.
- `runtime_sign_inventory.json`: actual loaded traffic actors, independent of XML signals.
- `episodes/*/three_cameras.mp4`: synchronized left/front/right H.264 video.
- `episodes/*/profile.json`: throughput and expert latency, with scope stated.
- `*.log`: CARLA recorder evidence; `carla.log`: simulator stdout.
- `error.json`/`stopped.json`: explicit failure or truncation reason.

A 200-step profile is deliberately truncated and cannot count as successful
arrival. Never mix it into final mission statistics. Zero collisions in a
profile, or even one completed route, is not a reliability claim. OpenDRIVE
signals are insufficient for rule coverage: the runtime contains 20 stop actors
in Town03 and 8 in Town04 despite their XML declaring none. Presence alone does
not establish an applicable encounter or compliant full stop on a given route.

Audit preserved evidence and compare two 200-frame same-seed runs:

```bash
.venv/bin/python scripts/assess_navigation.py --out outputs/navigation_m0_long_20261008
.venv/bin/python scripts/assess_navigation.py \
  --out outputs/navigation_m0_repeat_b_20261008 \
  --compare outputs/navigation_m0_repeat_a_20261008
```

New routes use the exact serialized original start/destination for planning.
Do not reuse the deprecated manifests listed in `configs/routes/README.md`.
`--rebuild-summary` preserves the initial derived summary and leaves raw records
untouched; it also fixes execution status for historical setup failures.

## Encounter inventory (rule applicability)

```bash
.venv/bin/python scripts/build_encounter_inventory.py \
  --out outputs/navigation_encounter_inventory_NEW
```

Starts one owned server per training town in turn, loads the map without a
Traffic Manager, and records the pinned evaluator's own red-light/stop-sign lane
applicability and every junction movement. A valid movement needs at least a 30 m
approach, and the harness planner must take it. Each town runs in a child
process, so a native client abort cannot orphan the server. It never drives,
never loads Town05 and freezes nothing. Current result:
`outputs/navigation_encounter_inventory_v2_20261009`.

## Development encounter catalog (frozen dev_v1)

`configs/encounters/dev_v1` holds 300 encounters (50 per family) over the five
training towns, selected by `scripts/build_encounter_catalog.py` with a fixed
seed, without driving outcomes. Do not rebuild or edit it: the builder refuses
to overwrite it, and `catalog.sha256` is checked by the runner. Scenarios:
NavigationSignalHold (held red, declared release rule), HardBreakRoute,
SignalizedJunctionLeftTurn and NonSignalizedJunctionLeft/RightTurn. Traffic seed
= `--seed` + route id.

```bash
systemd-inhibit --what=sleep:idle:handle-lid-switch --why=navigation \
  .venv/bin/python scripts/run_encounters.py \
  --out outputs/navigation_expert_ceiling_NEW --seed 42 --workers 3
python3 scripts/watch_navigation.py --out outputs/navigation_expert_ceiling_NEW
touch outputs/navigation_expert_ceiling_NEW/STOP   # clean stop; rerun the same command to resume
```

Chunks of 10 routes each get a fresh owned server. `--workers N` (at most
`runtime.max_servers`, currently 3) runs chunks on parallel slots: ports
2100/8100 + 100×slot, a separate lock per slot, and serialized server starts
that wait for at least 6 GB free VRAM and 12 GB available RAM (runner defaults).
Each launch waits until the requested map is loaded and driving has begun;
a watchdog pauses the newest chunk below 6 GB available RAM. The worker count
never changes chunk composition or seeds. Only a missing record or a
simulator crash is retried, once, and every attempt is kept. Headline per family:
`rule_safe` (completed with no collision, rule, route or blocking infraction).
Min-speed infractions and the strict any-infraction count are reported beside
it. Intervals must be clustered by junction/stop location before any claim.

The completed v2 continuation used two workers, with the same catalog/seeds and
record-preserving retry rules. Its user service is
`navigation-m0-v2-resume-20261011.service`; stop it through the queue's STOP file.
`navigation-m0-v2-assessment-20261011.service` waits for that runner's lock to
release and writes `reports/navigation_expert_ceiling_v2_20261009.json`.
Both services have now exited after 300/300 encounters. Do not resume this completed
queue. The corrected final analysis is
`reports/navigation_expert_ceiling_v2_audited_20261011.json`; see the
[M0 closeout](navigation_m0_closeout_20261011.md).
The Oct9 services stopped during host shutdown. These transient services do not
restart automatically after power-off; resume deliberately from preserved records.

```bash
python3 scripts/watch_navigation.py --out outputs/navigation_expert_ceiling_v2_20261009
.venv/bin/python scripts/assess_encounters.py \
  --out outputs/navigation_expert_ceiling_v2_20261009 \
  --report outputs/navigation_expert_ceiling_v2_20261009/live_assessment.json
```

The assessor distinguishes unrun encounters from failed/missing declared setups,
and binds each setup audit to the attempt that supplied its final driving result.
Traffic-light actor IDs are session-local: matching uses the hash-verified reference
inventory's physical position and entry lane, not equality of IDs across launches.
`all_collision_events` counts every raw attempt, including invalid setups and
retries; valid-final collisions are also shown separately. Family rates include
only valid settled trials, with all excluded/pending route IDs explicit. A null
cluster-bootstrap bound means insufficient/degenerate data, never zero risk.
Run process-aware live assessments on the host: a sandbox may not expose the
evaluator's PID. Active trials without records are shown as in progress rather
than charged as crashes. The queue bar updates when a chunk settles, so raw
records can be ahead of its displayed count.

## Current research boundary

M0's privileged teacher, cameras, evaluator and encounter calibration are complete.
It does not train a camera student or establish a JEPA navigation improvement.
M1's20k gate passed; M2's first camera student trained but failed blocked on its
long-route diagnostic. Read the current assessments linked below before continuing.
Multi-town probes and a 200-frame observable-world repeatability check now pass;
branching determinism is still unproven. Final Town05 routes must be frozen before model
selection. Multi-hour collection/training budgets are proposed from measured
throughput and require user approval as agreed.

## M1 collection

The [M1 pilot report](navigation_m1_pilot_20261015.md) documents the approved new
location split, atomic RGB shards, label boundaries, perturbations and measured
storage. The expert six-family pilot passed with 1,460 samples; the intentionally
capped noise test passed with 96 samples. The
[20k gate passed](navigation_m1_assessment_20261015.md); the100k gate remains pending.

The host reboot resolved the NVIDIA package mismatch: kernel and userspace both
report 615.78.08 on 2026-10-15. Simulation refuses an unavailable GPU-memory query;
do not bypass readiness checks.

The first collection stopped through STOP after 297 samples when the spatial guard
caught a spawn on a held-out junction boundary. Preserve that output; do not resume
it or use it for training. The frozen catalogs and held-out routes remain unchanged.
Collection now applies a geometry-only 1m training clearance filter: 287/300 routes
remain, covering every original family/town stratum. The strict runtime guard stays
active; a partition violation immediately fails the whole queue. See
[the collection note](navigation_m1_collection_20261015.md).

A fresh 20k collection ran from14:57 to15:39 IST under the persistent user service
`navigation-m1-20k-v2-20261015.service`, which has exited. The command below
identifies the completed request; do not restart it or launch a duplicate.

```bash
systemd-inhibit --what=sleep:idle:handle-lid-switch --why='M1 bounded collection' \
  .venv/bin/python -u scripts/collect_navigation.py \
  --out outputs/navigation_m1_20k_v2_20261015 --target-samples 20000 --max-wall 5400
python3 scripts/watch_navigation.py --out outputs/navigation_m1_20k_v2_20261015
touch outputs/navigation_m1_20k_v2_20261015/STOP
```

This invocation is bounded to 90 minutes plus
cleanup and performs a full data audit on exit. Resume only with the same source/
config request; explicitly clear STOP when resuming intentionally. Multi-hour
collection/training still requires the agreed approval. One simulator at a time;
minimum 6 GB free VRAM/12 GB available RAM at launch and 25 GB free disk preserved.

The above collection **completed at15:39:02 on2026-10-15**; do not restart it.
All20k timestamps passed pixel/action and independent spatial audits;63 episodes/
17,567 timestamps are eligible for imitation. All73 raw attempts, including three
collisions and the infrastructure/final-cap prefixes, remain preserved.

## M2 camera student

Read [the first student assessment](navigation_m2_student_smoke_20261015.md).
The first1,000-step job completed in441s, including preparation. The best policy
is step300 at `outputs/navigation_m2_student_smoke_20261015/best_policy.pt`.
Do not confuse `latest.pt` with the selected policy. Its original three training
sources are hash-verified under the run's `source/`; full optimizer/RNG/sampler
recovery state is in `latest.pt`. CUDA BF16 does not use a gradient scaler.

The13.605GB cache at `outputs/navigation_m2_cache_20k_20261015` stores raw resized
three-view RGB and original-resolution front RGB, not backbone features. It
contains the fixed internal location split:50/13 episodes,14,745/2,822 timestamps.
`heldout_dev` and Town05 are absent. Future cache reuse verifies provenance and
builder hashes; interrupted caches lack a completion manifest and are preserved.

Commands below identify completed requests and are examples for a reviewed NEW
output directory; do not rerun them into existing outputs:

```bash
.venv/bin/python scripts/train_navigation_student.py \
  --out outputs/navigation_m2_student_NEW --steps 1000 --max-wall 1800
python3 scripts/watch_navigation.py --out outputs/navigation_m2_student_NEW
touch outputs/navigation_m2_student_NEW/STOP
```

An interrupted training request resumes only with identical source/data/target
and explicit `--resume`. A completed request does not extend its step budget.
The trainer shares the single-navigation GPU lock with CARLA. Cache preparation
counts toward the bounded invocation. Disk reserve25GB; no multi-hour run without
the agreed approval.

Camera evaluation uses a separate inference process receiving only allowlisted
arrays/JSON. Simulator/teacher/collection imports are blocked there. Ground-truth
ego odometry/localization for map navigation is declared and matches collection;
other actor and ground-truth rule/hazard facts stay outside. This custom map/ego
experiment is not an official sensors-track result.

```bash
.venv/bin/python scripts/run_navigation.py student \
  --checkpoint outputs/navigation_m2_student_smoke_20261015/best_policy.pt \
  --out outputs/navigation_m2_camera_NEW --max-wall 600
python3 scripts/watch_navigation.py --out outputs/navigation_m2_camera_NEW
touch outputs/navigation_m2_camera_NEW/STOP
```

`--max-steps 200` is an intentionally truncated profile. `--rule-overrides`
explicitly enables the optional learned-perception rule layer; default OFF.
Each evaluation copies/hashes its checkpoint and archives code/config. Preserve
all outcomes. The current full-route diagnostic failed blocked at1.91%, not safe
completion. Its video and audit evidence are in
`outputs/navigation_m2_camera_route_20261015`; keep those recordings evaluation-only.
All current services are idle.

Next work: separate semantic stop intent from cruising speed and overspeed braking;
improve signal supervision and streaming memory; collect DAgger corrections on
eligible TRAIN routes. The uncorrected centreline is a required path baseline.
Calibrate the untouched heldout_dev suite before comparing students on it. The
Town05 suite was not registered before the first student as intended: disclose
that timing deviation and freeze geometry/coverage before any Town05 testing.

Continuation2026-10-16: [semantic-stop countermeasure](navigation_m2_stop_go_20261016.md)
trained a fresh2,000-step candidate on the same cache/split, with32-frame history,
batch2 and LR1e-4. Best internal checkpoint is step1,500. False holds improve,
required-stop recall does not; no driving improvement claim yet. The architecture
is selected from the checkpoint at inference; original v0 checkpoints still load.
Training archives sources at launch and resume refuses changed code/request.

To reproduce the request, use a reviewed fresh output (the actual run completed):

```bash
.venv/bin/python scripts/train_navigation_student.py \
  --out outputs/navigation_m2_stop_go_NEW --steps 2000 --max-wall 1800 \
  --variant stop_go --learning-rate 0.0001 --batch-size 2 --clip-length 32
.venv/bin/python scripts/assess_navigation_student.py \
  --out outputs/navigation_m2_stop_go_NEW \
  --reference outputs/navigation_m2_student_smoke_20261015 \
  --clip-length 32 --device cuda --report outputs/navigation_m2_stop_go_NEW/assessment.json
```

The frozen small TRAIN pilot is in
`reports/navigation_m2_stop_go_pilot_20261016.json`; all six cases/seeds/candidates
remain fixed. It is not heldout_dev. Its first attempt was preserved after a
recording bug; the corrected retry uses unique native files under each run's
`recordings/`, separate from console text. Multi-route native recordings made
before this fix were overwritten per invocation; existing per-episode MP4s,
world traces and raw scores remain separate. Check current pilot status in the
handoff and live watcher before launching anything.

## M2 correction collection and protected fine-tuning

Read [the DAgger result](navigation_m2_dagger_20261016.md). The completed pilot is
`outputs/navigation_m2_dagger_pilot_20261016`; the completed fine-tune is
`outputs/navigation_m2_dagger_finetune_20261016`. Do not restart either.
Data passed its gate, but no trained checkpoint beat the parent. The selected
step0 is the old v2 model; latest1000 is rejected, not a new driving candidate.

`dagger` is a distinct training command. It requires a frozen train catalog,
explicit eligible route IDs, a source checkpoint and a guard preserving the old
internal validation locations. It never enables privileged control in `student`
evaluation. Parent-side assistance and video labels explicitly mark the mixture.
Never report these assisted arrivals as camera-policy performance.

Examples below require fresh output/cache paths and a bounded reviewed request:

```bash
.venv/bin/python scripts/collect_navigation_corrections.py prepare \
  --guard outputs/correction_guard_NEW.json
.venv/bin/python scripts/collect_navigation_corrections.py pilot \
  --guard outputs/correction_guard_NEW.json --out outputs/correction_pilot_NEW --max-wall 900
python3 scripts/watch_navigation.py --out outputs/correction_pilot_NEW
.venv/bin/python scripts/assess_navigation_corrections.py \
  --out outputs/correction_pilot_NEW --decode-videos \
  --report outputs/correction_pilot_NEW/correction_assessment.json
```

The assessment gate must pass before any correction cache/training. Censored
episodes never enter clean imitation; only the separate expert-correction reader
may expose their instantaneous labels. All assisted future-speed masks remain0.
Keep the original cache and validation unchanged; the new raw cache contains only
correction samples. Fresh fine-tunes use `--init-from` with a new AdamW, not resume;
parent weights are validated/exported at step0 and remain selected if all updates
are worse. Optimizer/RNG/sampler and both dataset manifests are saved.

```bash
.venv/bin/python scripts/train_navigation_student.py \
  --variant stop_go --init-from outputs/navigation_m2_stop_go_20261016/best_policy.pt \
  --corrections outputs/correction_pilot_NEW/correction_assessment.json \
  --correction-cache outputs/correction_cache_NEW --out outputs/correction_finetune_NEW \
  --steps 1000 --max-wall 1200 --learning-rate 0.00005 --batch-size 2 --clip-length 32
```

This is syntax documentation, not a recommendation to repeat the rejected run.
Next investigate localized rule labels/visibility and long stop/release context.
No multi-hour collection/training or benchmark/reward change without user approval.

## M2 rule-observability diagnostics

Read [the spatial/temporal result](navigation_m2_rule_observability_20261016.md)
and the [completed pixel/perception continuation](navigation_m2_signal_perception_20261016.md).
The old stopped diagnostic server has exited after approved graceful cleanup;
no cleanup approval is pending. Inventories, static labels and component models
are evidence, not a promoted driving candidate. Check live state before a server.

These commands document the diagnostic interfaces; every simulator/audit invocation
requires a fresh output. Static capture accepts approved training towns only.
Projection audit is scoped to the existing audited Town03 correction pilot.

```bash
.venv/bin/python scripts/capture_navigation_rule_geometry.py \
  --town Town03_Opt --max-wall 240 --out outputs/rule_geometry_NEW
.venv/bin/python scripts/audit_navigation_rule_observability.py \
  --geometry outputs/rule_geometry_NEW --out outputs/rule_audit_NEW
.venv/bin/python scripts/probe_navigation_signal_pixels.py \
  --audit outputs/rule_audit_NEW --geometry outputs/rule_geometry_NEW \
  --out outputs/signal_pixels_NEW
```

The last command has a240s wall cap plus bounded cleanup. It renders red/green/red
at fixed TRAIN poses with dynamic traffic omitted. It neither trains nor evaluates
a driving policy. All paired images, exact camera-matrix checks, repeated-red
control, source hashes and setup failures must be retained. Frustum overlap and
pixel response do not certify original-scene visibility or approve state labels.

## M2 static signal supervision and component diagnostics

The successful static corpus is `outputs/navigation_m2_signal_corpus_v2_20261016`:
51 poses/612 images/63 supported head-view pairs, with a passed five-town gate.
The earlier `outputs/navigation_m2_signal_corpus_retry2_20261016` gate remains
failed; never use it for fitting. The valid corpus uses
`reports/navigation_m2_signal_pairs_plan_v3_20261016.json` and the frozen
[response-validator protocol](navigation_m2_signal_pairs_protocol_v2_20261016.md).
All locations are TRAIN, excluding the original internal driving validation too.
Recovery after native failure verifies original worker/source/plan and completed
image hashes, skips complete towns and archives the failed town. Do not resume
or recollect this completed corpus. STOP is `touch <output>/STOP`.

Both `outputs/navigation_m2_signal_grid_20261016` and
`outputs/navigation_m2_signal_grid_region_20261016` completed1,000 updates and
failed component diagnostics. Their `best_component.pt` files are perception-only,
not camera-policy checkpoints. The oracle crop classifier uses privileged boxes;
the separate reference detector uses declared COCO supervision and misses most
supported small lights. No component is integrated or approved by driving results.

Rebuild the comparison/graphs without training or CARLA:

```bash
.venv/bin/python scripts/assess_navigation_signal_perception.py
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_corpus_v2_20261016 --once
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_grid_region_20261016 --once
```

For future bounded component runs, `collect_navigation_signal_pairs.py --help`,
`resume_navigation_signal_pairs.py --help` and
`train_navigation_signal_grid.py --help` document the interfaces. Freeze a fresh
plan/protocol before collecting or fitting; preserve all failed attempts and
whole-junction evaluation masks. Do not repeat the rejected fit for longer.

## M2 small-anchor detector and crop alignment

Read [the selected component result](navigation_m2_signal_detector_20261016.md).
The completed200-update detector run is
`outputs/navigation_m2_signal_detector_20261016`; selected50 `best_component.pt`
is perception-only. Latest200 has higher sparse recall but is not selected by the
frozen loss. Do not restart this run or replace camera `best_policy.pt` with it.
`outputs/navigation_m2_signal_crop_alignment_20261016/robust_rgb_linear.npz`
improves state recognition on the same predicted boxes; oracle68/72 is a separate
privileged diagnostic, while combined selected FP32 result is24/72.

Full FP16 remains above100ms/model triplet. The200-proposal variant reaches85.6ms
but loses7 light detections; it is rejected as default. Both inference-only audits
are complete and archived. Keep every threshold/coverage/latency tradeoff visible.

Rebuild the aggregate without fitting, GPU or changing its experiment inputs:

```bash
.venv/bin/python scripts/summarize_navigation_signal_detector.py
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_detector_20261016 --once
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_proposal_budget_20261016 --once
```

Future interfaces: `train_navigation_signal_detector.py --help`,
`train_navigation_signal_crops.py --help`, `assess_navigation_signal_precision.py --help`.
Freeze a new protocol/output before experiments. Detector checkpoints include
full optimizer/RNG; current code also records validation_step for interrupted
scoring. Resume checks original request/archive/current source; the older completed
run predates that bookkeeping correction and should not be resumed with new code.
Do not rewrite its one_to_one_audit.json: the crop experiment hashes that input.
Next data/negative mining uses only component-TRAIN groups; preserve all validation
locations/results and the held-out geographical guards.

## M2 verified negatives and rejected refinement

Read [the completed result](navigation_m2_signal_hard_negatives_20261016.md).
110 canonical negatives passed registered TRAIN-only semantic/RGB checks; use
`outputs/navigation_m2_signal_negative_release_20261016/release.json` explicitly.
The original mining `training_release.json` remains false as the preserved initial
veto. A successful automatic mining gate alone never authorizes these labels.

The fresh fit `outputs/navigation_m2_signal_hard_negative_fit_retry_20261016`
completed200 updates. Its selected0 equals the original step50 detector exactly;
**latest200 is rejected** (27/72 detections vs30/72 parent), despite fewer
approximate backgrounds. Do not restart either completed run or the stopped first
attempt. Calibration `outputs/navigation_m2_signal_train_calibration_20261016`
chooses0 bias and reuses latest200's exact evaluation; no calibrated checkpoint.
No driver was changed. Current inference components remain insufficient for M2.

```bash
.venv/bin/python scripts/summarize_navigation_signal_negatives.py
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_hard_negative_fit_retry_20261016 --once
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_train_calibration_20261016 --once
```

The aggregate only verifies inputs and rebuilds reports/figures. Future interfaces:
`mine_navigation_signal_negatives.py --help`,
`verify_navigation_signal_negatives.py --help`,
`calibrate_navigation_signal_head.py --help`. Use fresh outputs/frozen protocols;
STOP remains `touch <output directory>/STOP`. First broaden positive TRAIN
approaches/views; preserve all validation locations, no threshold search or blind
long repeat. Tracking, lane applicability and full stop/release memory remain.

## M2 TRAIN approach expansion and completed joint fit

Read [the completed result](navigation_m2_signal_expansion_20261016.md).
Data `outputs/navigation_m2_signal_expansion_20261016` passed99 poses/1188 images,
116 responsive pairs. Explicit TRAIN-only append preserves original144 validation
images/72 labels byte-for-byte; never call the seeded splitter on this subset.
TRAIN positive phase labels total642. Old verified440 negative copies retained.

`outputs/navigation_m2_signal_joint_expansion_20261016` completed200 updates.
Selected0 equals original detector50. Latest200 improves detection36/72, joint26,
approximate backgrounds209 and proposals69/72 but is unselected by higher fixed
loss. Do not resume completed runs, relabel latest as selected or confuse it with
the earlier rejected head-only latest. Descriptive loss/sampler audits are complete
in `outputs/navigation_m2_signal_joint_loss_audit_20261016` and
`outputs/navigation_m2_signal_joint_sampling_audit_20261016`.

```bash
.venv/bin/python scripts/summarize_navigation_signal_expansion.py
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_joint_expansion_20261016 --once
```

Future interfaces: `prepare_navigation_signal_expansion.py --help`,
`collect_navigation_signal_pairs.py --help`,
`train_navigation_signal_detector.py --help` (explicit `--supplement`),
`assess_navigation_signal_losses.py --help`. New protocols/fresh outputs precede
any future experiment. No retroactive selection rule or blind longer fit.

## M2 ROI residual and selected perception bundle

Read [the completed component result](navigation_m2_signal_roi_adapter_20261016.md).
`outputs/navigation_m2_signal_roi_adapter_20261016/best_candidate.pt` selects500
under its prospective operational guard:44/72 detected,29 old-state-correct,
160 approximate backgrounds. Latest1000 loses Town03 coverage and is rejected.
Do not resume it or restart completed cache extraction. Initialization came from
the explicitly unselected joint200; previous selections remain unchanged.

Separate `outputs/navigation_m2_signal_state_expansion_20261016` fits one C1 RGB
state classifier and passes:31/72 joint on identical44 detections; old30-box
compatibility27/30; oracle66/72 is privileged and regresses from68. Its
`perception_bundle.json` binds selected detector/state hashes. This is NOT a
driving checkpoint or automatic policy promotion.

The detector architecture is `signal-detector-roi-residual-v1`; construct
`make_signal_detector(roi_adapter=True)` then strict-load its model state. Only
65,665 adapter parameters changed; all joint base/proposal/box-regression weights
are identical. Bundle state NPZ uses the same crop_feature/linear prediction API.

```bash
.venv/bin/python scripts/summarize_navigation_signal_roi_adapter.py
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_roi_adapter_20261016 --once
python3 scripts/watch_navigation.py --out outputs/navigation_m2_signal_state_expansion_20261016 --once
```

New interfaces: `train_navigation_signal_roi_adapter.py --help`,
`train_navigation_signal_state_expansion.py --help`. Fresh protocol/output required;
no source-changing resume is implemented for these completed bounded fits. STOP
remains the output-directory file. Next tracking/applicability/complete stop-release
observability on existing TRAIN rollouts; no component-validation mining or blind
longer fit. No new driving result until those checks and a matched pilot.
