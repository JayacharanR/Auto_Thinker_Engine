# Frozen pretrained proposal-verification experiment

Register before feature extraction/fitting. Previous context/data/hard-negative
experiments remain failed. Test a semantic visual prior and binary foreground
verification while retaining the original RGB proposer's phase estimates.
This changes representation, classification task and TRAINdiversity; it is an
engineering countermeasure, NOT an isolated claim about ImageNet pretraining.

Use original selected bulb proposal checkpoint
27ddcc86e6be56a2edc7033e5ac9b47100f0c305d86e34893ced3cb20000b193,
fixed.5threshold/8connectedcomponents, alloriginal38,543VALproposals and72known
housing denominator. Same144VALimages/labels/ignore masks, unchanged source roles.
TRAIN2844images from21eligiblejunctions: exact164case supplement plus old876.
All36unknown supplement cases remain excluded; Town10signalcoverage gap remains.

TRAINmining same prior deterministic rule: allknownpositiveproposals plus≤4audited
active-bulb anchors/housing; at most16known-background proposals/image (fourhighest
score,12seededrandom). Unknown/ambiguousprojectedlightregions excluded. NoVALmining.
Cached patches contain onlynative32RGB and96RGBcontext averagedto32, centered
on a predictedpoint. Noabsolutecoordinate/source/phase-label/map input toclassifier.

Twofixedbinarylogisticheads, balancedclasses,C1,L2,lbfgs,max_iter1000,tol1e-4,
TRAIN-onlyStandardScaler and fixed.5foregroundprobability threshold; no sweeps,
validation fitting, head selection or fit continuation. Negative0/positive1.
1. Low-costRGBbaseline: nativecenter5x5RGB(75numbers), localRGBmean/std(6),
   contextRGBmean/std(6),87features, allscaled0..1.
2. FrozenImageNetResNet18: sourceweights SHA
   f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec,
   noCARLAlabeltraining. Each32patch bilinearresize128, ImageNetmean/std,
   evalmodeBF16. Localcentral2x2finalfeaturecellsmean512 pluscontextglobalmean512:
   1024features, cachedFP16. Freezeallencoderweights/buffers. Nofine-tuning.

EveryVALproposal gets a score before labels are consulted. Retainedpointskeep
originalproposerstate; rankbyverifierforegroundprobability, fixedpoint-in-box
matching. Reportbothheadsagainstoriginal56localized/44joint/37241backgroundpeaks,
alltown/statecounts and everyunknownregion. Neither canrecovermissingproposals.
Protection:joint≥44,town≥7/15/12/10,state≥22red/15yellow/7green; usefulcomponent
also requires≥90%fewerknown-backgroundpeaks. No thresholdorclassgaterelaxation.
No automaticdeployment/nativepilot or learnedruleclaim. M4primaryunchanged.

Persist compressedfeaturecache/TRAINrows/proposaloffsets/sourcehashes/fittedscaler
andcoefficients/probabilities. NoextraRGBcopy. Fixedencoder/outputparitychecked
beforeexport; nonconvergence retainedandtreatedasfailedfit. CPU2threads.
Featurestage1800s,eachhead600s,whole3000s;400MBartifactcap, originalguardian
immutable14GBnet/10GBmin andSTOP. Standardfitcalls are bounded inside an owned
child if needed; noSIGKILL. Fullruntime latency remainsunqualified until measured.
