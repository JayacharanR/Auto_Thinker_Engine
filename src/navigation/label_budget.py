"""Prospective nested whole-group label budgets, without validation resplitting."""

import hashlib
import json

import numpy as np


def nested_budgets(groups, fractions=(0.05, 0.1, 0.25, 1.0), seed=42):
    """Use one seeded order and maximum-budget prefixes; report actual fractions.

    A group is a complete driving episode or all phases/views of a labeled
    static pose. Use independent ledgers for each annotation modality; never
    hide a fully labeled common component inside a low-label driving arm.
    """
    if not groups or len({g["id"] for g in groups}) != len(groups):
        raise ValueError("Nonempty uniquely identified whole annotation groups required")
    if list(fractions) != sorted(set(fractions)) or min(fractions) <= 0 or max(fractions) != 1:
        raise ValueError("Increasing positive nested fractions must end at100%")
    for group in groups:
        if (
            group.get("role") != "train"
            or group.get("partition") != "train"
            or group["town"].removesuffix("_Opt") == "Town05"
        ):
            raise ValueError("Only released TRAIN annotations may consume the label budget")
        if not isinstance(group["labels"], int) or group["labels"] <= 0:
            raise ValueError("Each whole group needs an explicit positive annotation count")
    ordered = sorted(groups, key=lambda g: g["id"])
    rng = np.random.default_rng(seed)
    ordered = [ordered[int(i)] for i in rng.permutation(len(ordered))]
    cumulative = np.cumsum([g["labels"] for g in ordered])
    total = int(cumulative[-1])
    allocations = []
    for fraction in fractions:
        budget = int(total * fraction + 1e-9)
        length = int(np.searchsorted(cumulative, budget, side="right"))
        selected = ordered[:length]
        count = sum(g["labels"] for g in selected)
        allocations.append(
            dict(
                nominal_fraction=fraction,
                maximum_labels=budget,
                actual_labels=count,
                actual_fraction=count / total,
                groups=[g["id"] for g in selected],
                empty=not selected,
            )
        )
    encoded = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    return dict(
        schema="whole-group-label-budget-v1",
        seed=seed,
        annotation_groups_sha256=hashlib.sha256(encoded).hexdigest(),
        total_labels=total,
        fixed_order=[g["id"] for g in ordered],
        allocations=allocations,
        note=(
            "Actual x-axis fractions may be below nominal budgets; "
            "no partial episodes or resampling for coverage."
        ),
    )
