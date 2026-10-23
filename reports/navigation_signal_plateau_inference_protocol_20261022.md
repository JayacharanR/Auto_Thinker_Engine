# Complete TRAIN classification of independent RGB plateau candidates

Registered before model inference. The closed plateau inventory completes63.69s,
12,362,313unionpoints, including every1,770,643previousrefinedpoint. Candidate
coverage1,209→1,246/1,248housinglabels;37previouslyuncoveredhousingsgainpoints,
43of59misseshaveexactrecognizedoraclecoordinatesinthenewset. These are coverage
diagnostics, not new classifier predictions. Original dense15Mcapfailure remains.

Classify EVERY saved new coordinate with the unchanged supported head whose TRAIN
fit excluded the image's wholejunction. Reuse exact previous refined probabilities
for all old points. Recompute all48,865originalsampledRGBfeatures and OOFprobabilities
for exactparity. Same87RGBfeatures/FP16cacheboundary, original.5foregroundthreshold,
phaseargmax, scoreordering, all1,248labels and background/unknown masks. No fitting,
selection sweep, classifierweightchange, benchmarkchange or componentVAL.

Predeclared continuation: correctjoint strictlyexceeds1,189, everytown/phasejoint
doesnotregress versus localrefinement, and knownbackground≤3,075.8(90%reduction
versusrawRGBcontrol30,758). Report unknowns and all candidate/timing costs. If any
criterionfails, retainresult/no componentcheck. Allold/newcoordinates and allfour
probabilities retained, plusper-image partialmetricreceipts. No truncation on budget.

Output outputs/navigation_signal_plateau_train_20261022. CPU2threads,3600s whole
stagecap,600MB artifactcap. Expected probabilityarray395.6MB plus49.4MBcoordinates;
alloldscientificartifacts retained. Original14GBnet/10GBminimum storageguardian
applies; availableheadroommustbecheckedbeforelaunch. STOP/no retries. Source bytes
pinnedthroughclosure; noglobalproposalrulechangeinflight. Incrementalclassification
timings exclude priorproposer/generation/scoring andarenotafullruntimeprofile.

No native driving or automatic deployment. Latest component55/72stillfailsyellow;
M2/M3/M5remainunqualified,M6notrun. Thisdevelopmentstudydoesnotreplacefinaldriving.
