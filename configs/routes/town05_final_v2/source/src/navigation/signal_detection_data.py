"""Training-only detection boxes; original static dataset/split are immutable."""

import json
import math
from collections import defaultdict

import numpy as np
import torch
import yaml

from src.navigation.dagger import digest
from src.navigation.rule_geometry import project_box
from src.navigation.sensors import calibrations
from src.navigation.signal_learning import SignalDataset


class SignalDetectionData:
    def __init__(self, corpus, *, training_only=False):
        self.base = SignalDataset(corpus, training_only=training_only)
        self.root = self.base.root
        audit = json.loads((self.root / "data_audit.json").read_text())
        self.labels = defaultdict(list)
        for label in audit["labels"]:
            if label["static_phase_response_supported"]:
                self.labels[label["town"], label["case_id"], label["view"]].append(label)
        calibration = calibrations(
            yaml.safe_load((self.root / "source/configs/navigation.yaml").read_text())
        )
        ignore = {}
        for town in sorted({s["town"] for s in self.base.samples}):
            geometry = json.loads((self.root / town / "geometry.json").read_text())
            cases = json.loads((self.root / town / "cases.json").read_text())
            for case in cases:
                for view, cal in calibration.items():
                    boxes = []
                    for actor in geometry["lights"]:
                        for housing in actor["boxes_world"]:
                            projected = project_box(housing["vertices"], case["ego_pose"], cal)
                            if projected["in_image"]:
                                x0, y0, x1, y1 = projected["bbox_xyxy"]
                                box = [
                                    max(0, 2 * math.floor(x0 / 2) - 4),
                                    max(0, 2 * math.floor(y0 / 2) - 4),
                                    min(512, 2 * math.ceil(x1 / 2) + 4),
                                    min(288, 2 * math.ceil(y1 / 2) + 4),
                                ]
                                if box[2] > box[0] and box[3] > box[1]:
                                    boxes.append(box)
                    ignore[town, case["route_id"], view] = np.array(boxes, np.float32).reshape(
                        -1, 4
                    )
        self.targets, self.annotation_rows = [], []
        for sample in self.base.samples:
            key = sample["town"], sample["case_id"], sample["view"]
            labels = [
                label
                for label in self.labels[key]
                if sample["phase"] != "yellow" or label["yellow_color_supported"]
            ]
            boxes = np.array([label["crop_xyxy"] for label in labels], np.float32).reshape(-1, 4)
            if len(boxes) != len(sample["points"]):
                raise ValueError("Detection denominator differs from fixed center evaluation")
            self.targets.append(
                {
                    "boxes": boxes,
                    "labels": np.ones(len(boxes), dtype=np.int64),
                    "ignore_boxes": ignore[key],
                }
            )
            self.annotation_rows.append(labels)
        assert sum(len(self.targets[i]["boxes"]) for i in self.base.validation) == (
            0 if training_only else 72
        )
        self.positive_train = [i for i in self.base.train if len(self.targets[i]["boxes"])]

    def batch(self, indices, device):
        rgb = [
            torch.from_numpy(self.base.images[i].copy()).to(device).float() / 255 for i in indices
        ]
        targets = [
            {k: torch.from_numpy(v.copy()).to(device) for k, v in self.targets[i].items()}
            for i in indices
        ]
        return rgb, targets

    def sample_train(self, rng):
        if hasattr(self, "training_domains"):
            original, supplement = self.training_domains
            positive_groups = self.positive_domains
            return [
                int(rng.choice(positive_groups[int(rng.integers(2))])),
                int(rng.choice((original, supplement)[int(rng.integers(2))])),
            ]
        # Fixed mixture: one image with supported housing, one uniform TRAIN image.
        return [int(rng.choice(self.positive_train)), int(rng.choice(self.base.train))]

    def append_training_corpus(self, directory):
        """Append only approved original TRAIN groups; validation bytes stay fixed."""
        from pathlib import Path

        directory = Path(directory)
        if hasattr(self, "training_domains"):
            raise ValueError("Only one frozen supplement per experiment")
        extra = SignalDetectionData(directory, training_only=True)
        plan = json.loads((directory / "plan.json").read_text())
        if plan["expansion"]["base_audit_sha256"] != digest(self.root / "data_audit.json"):
            raise ValueError("Supplement belongs to a different original corpus")
        if any(self.base.roles.get(group) != "train" for group in extra.base.roles):
            raise ValueError("Supplement overlaps validation or unknown component groups")
        original_config = yaml.safe_load((self.root / "source/configs/navigation.yaml").read_text())
        extra_config = yaml.safe_load((directory / "source/configs/navigation.yaml").read_text())
        if calibrations(original_config) != calibrations(extra_config):
            raise ValueError("Supplement changed native camera rig")
        if extra.base.validation or any(s["role"] != "train" for s in extra.base.samples):
            raise ValueError("Non-TRAIN supplement image")
        if self.labels.keys() & extra.labels.keys():
            raise ValueError("Supplement case identities collide")
        validation = self.base.validation.copy()
        old_rgb = self.base.images[validation].copy()
        offset = len(self.base.samples)
        original_train = self.base.train.copy()
        original_positive = self.positive_train.copy()
        new_train = [offset + i for i in extra.base.train]
        new_positive = [offset + i for i in extra.positive_train]
        self.base.images = np.concatenate([self.base.images, extra.base.images])
        self.base.samples.extend(
            {**s, "supplement_corpus": str(directory.resolve())} for s in extra.base.samples
        )
        self.base.train.extend(new_train)
        for target in extra.targets:
            target["hard_negative_boxes"] = np.empty((0, 4), np.float32)
        self.targets.extend(extra.targets)
        self.annotation_rows.extend(extra.annotation_rows)
        self.labels.update(extra.labels)
        self.positive_train.extend(new_positive)
        self.training_domains = original_train, new_train
        self.positive_domains = original_positive, new_positive
        assert self.base.validation == validation
        assert np.array_equal(self.base.images[validation], old_rgb)
        assert sum(len(self.targets[i]["boxes"]) for i in validation) == 72
        provenance = dict(
            corpus=str(directory.resolve()),
            audit_sha256=digest(directory / "data_audit.json"),
            plan_sha256=digest(directory / "plan.json"),
            added_train_images=len(new_train),
            added_positive_labels=sum(len(t["boxes"]) for t in extra.targets),
            validation_images_unchanged=len(validation),
            validation_labels_unchanged=72,
            sampling="50/50 original/supplement independently for positive/uniform slots",
        )
        self.base.manifest["train_supplement"] = provenance
        return provenance

    def attach_hard_negatives(self, directory, expected_parent, release_path=None):
        """Validate every index/role/source before adding TRAIN-only replay targets."""
        from pathlib import Path

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        request = json.loads((directory / "request.json").read_text())
        release_path = (
            Path(release_path) if release_path is not None else directory / "training_release.json"
        )
        release = json.loads(release_path.read_text())
        if not release["approved"] or release["manifest_sha256"] != digest(
            directory / "manifest.json"
        ):
            raise ValueError(
                "Semantic training release required; automatic coverage is insufficient"
            )
        if not manifest["data_gate_passed"] or manifest["parent_sha256"] != expected_parent:
            raise ValueError("Passed hard-negative gate and fixed parent required")
        if manifest["corpus_audit_sha256"] != digest(self.root / "data_audit.json"):
            raise ValueError("Mined negative source corpus changed")
        for path, expected in request["source_hashes"].items():
            if digest(directory / "source" / path) != expected:
                raise ValueError("Mined negative worker archive changed")
        if not release.get("semantic_evidence_sha256"):
            raise ValueError("Registered semantic evidence required")
        for path, expected in release["source_hashes"].items():
            if digest(release_path.parent / "source" / path) != expected:
                raise ValueError("Semantic verifier archive changed")
        for path, expected in release["semantic_evidence_sha256"].items():
            if digest(Path(__file__).resolve().parents[2] / path) != expected:
                raise ValueError("Semantic evidence changed")
        approved = set(release["approved_candidate_indices"])
        if len(approved) != len(release["approved_candidate_indices"]):
            raise ValueError("Duplicate semantic approval index")
        pending, counter, groups, towns = {}, 0, set(), set()
        for row in manifest["rows"]:
            selected = [n for i, n in enumerate(row["negatives"], counter) if i in approved]
            counter += len(row["negatives"])
            if not selected:
                continue
            groups.add((row["sample"]["town"], row["sample"]["junction_id"]))
            towns.add(row["sample"]["town"])
            for phase, index in row["phase_indices"].items():
                sample = self.base.samples[index]
                if (
                    sample["role"] != "train"
                    or sample["path"] != row["phase_paths"][phase]
                    or (sample["town"], sample["case_id"], sample["view"])
                    != (row["sample"]["town"], row["sample"]["case_id"], row["sample"]["view"])
                ):
                    raise ValueError("Mined negative validation/identity contamination")
                boxes = np.asarray([n["box"] for n in selected], np.float32).reshape(-1, 4)
                if not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
                    raise ValueError("Invalid mined negative box")
                if index in pending:
                    raise ValueError("Repeated mining sample")
                pending[index] = boxes
        if (
            any(i < 0 or i >= counter for i in approved)
            or len(approved) < 20
            or len(groups) < 6
            or len(towns) < 3
        ):
            raise ValueError("Semantic subset fails predeclared coverage gate")
        for index, target in enumerate(self.targets):
            target["hard_negative_boxes"] = pending.get(index, np.empty((0, 4), np.float32))
        return {
            "manifest_sha256": digest(directory / "manifest.json"),
            "release_sha256": digest(release_path),
        }
