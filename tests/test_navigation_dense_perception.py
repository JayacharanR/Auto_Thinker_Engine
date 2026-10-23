"""Visible-mask denominators and one-to-one matching cannot inflate detector coverage."""

import numpy as np

from scripts.assess_navigation_dense_perception import match_boxes, visible_objects


def test_bipartite_matching_maximizes_count_not_greedy_iou(monkeypatch):
    import scripts.assess_navigation_dense_perception as audit

    # Greedy.9would leave the second reference unmatched; the valid pairing is.4/.8.
    monkeypatch.setattr(audit, "box_iou", lambda a, b: np.array([[0.9, 0.4], [0.8, 0.2]]))
    objects = [dict(group="vehicle", box=[0, 0, 10, 10]), dict(group="vehicle", box=[0, 0, 10, 10])]
    assert {(m["reference"], m["prediction"]) for m in match_boxes(objects, objects)} == {
        (0, 1),
        (1, 0),
    }
    # Restore a shape-appropriate scalar case: one box never counts as two actors.
    monkeypatch.setattr(audit, "box_iou", lambda a, b: np.full((len(a), len(b)), 0.9))
    assert len(match_boxes(objects, objects[:1])) == 1
    person = [dict(group="person", box=[0, 0, 10, 10])]
    assert match_boxes(objects, person) == []


def test_visible_instances_merge_semantic_parts_and_preserve_tiny_denominator():
    raw = np.zeros((288, 512, 4), np.uint8)
    depth = np.full((288, 512), 10.0, np.float32)
    raw[10:12, 20:24, 1] = 2
    raw[10:12, 20:24, 0] = 1
    raw[10:12, 20:24, 2] = 14
    raw[11, 20:24, 2] = 15  # One render actor can have different vehicle semantics.
    raw[20, 40:47, 1] = 3
    raw[20, 40:47, 2] = 12
    objects, tiny = visible_objects(raw, depth)
    assert len(objects) == 1 and len(tiny) == 1
    assert objects[0]["visible_pixels"] == 8 and objects[0]["box"] == [20, 10, 24, 12]
    assert objects[0]["semantic_classes"] == [14, 15] and objects[0]["render_actor_low16"] == 258
    assert tiny[0]["visible_pixels"] == 7 and tiny[0]["group"] == "person"
