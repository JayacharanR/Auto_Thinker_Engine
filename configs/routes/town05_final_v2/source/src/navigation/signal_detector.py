"""RGB-only housing detector; privileged ignore regions exist only during fitting."""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes
from torchvision.ops import box_area


def ignore_negatives(boxes, labels, ignored):
    """Keep positives; ignore negatives overlapping uncertain geometry (IoMin>=.3)."""
    result = labels.clone()
    if not len(boxes) or not len(ignored):
        return result
    lower = torch.maximum(boxes[:, None, :2], ignored[None, :, :2])
    upper = torch.minimum(boxes[:, None, 2:], ignored[None, :, 2:])
    intersection = (upper - lower).clamp(min=0).prod(2)
    denominator = torch.minimum(box_area(boxes)[:, None], box_area(ignored)[None, :])
    overlap = intersection / denominator.clamp(min=1e-8)
    result[(labels == 0) & (overlap.amax(1) >= 0.3)] = -1
    return result


class IgnoreTransform(GeneralizedRCNNTransform):
    def resize(self, image, target=None):
        original_size = image.shape[-2:]
        regions = {
            key: target[key]
            for key in ("ignore_boxes", "hard_negative_boxes")
            if target is not None and key in target
        }
        image, target = super().resize(image, target)
        for key, boxes in regions.items():
            target[key] = resize_boxes(boxes, original_size, image.shape[-2:])
        return image, target


class IgnoreRPN(RegionProposalNetwork):
    def assign_targets_to_anchors(self, anchors, targets):
        labels, matched = super().assign_targets_to_anchors(anchors, targets)
        return [
            ignore_negatives(a, label, target["ignore_boxes"])
            for a, label, target in zip(anchors, labels, targets)
        ], matched


class IgnoreRoIHeads(RoIHeads):
    def select_training_samples(self, proposals, targets):
        self._ignore_regions = [t["ignore_boxes"] for t in targets]
        try:
            result = super().select_training_samples(proposals, targets)
        finally:
            del self._ignore_regions
        sampled, matched, labels, regression = result
        regression = list(regression)
        for i, target in enumerate(targets):
            hard = target.get("hard_negative_boxes")
            if hard is None or not len(hard):
                continue
            positive = target["boxes"]
            extra = torch.cat([positive, hard]).repeat(4, 1)
            extra_labels = torch.cat(
                [
                    torch.ones(len(positive), dtype=torch.long, device=extra.device),
                    torch.zeros(len(hard), dtype=torch.long, device=extra.device),
                ]
            ).repeat(4)
            sampled[i] = torch.cat([sampled[i], extra])
            matched[i] = torch.cat(
                [matched[i], torch.zeros(len(extra), dtype=torch.long, device=extra.device)]
            )
            labels[i] = torch.cat([labels[i], extra_labels])
            # GT proposals have zero regression targets; negative targets are unused.
            regression[i] = torch.cat([regression[i], extra.new_zeros((len(extra), 4))])
        return sampled, matched, labels, regression

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched, labels = super().assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        return matched, [
            ignore_negatives(p, label, ignored)
            for p, label, ignored in zip(proposals, labels, self._ignore_regions)
        ]


class ResidualSignalPredictor(torch.nn.Module):
    """Bounded nonlinear foreground correction; frozen base/box outputs retained."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(base.cls_score.in_features, elementwise_affine=False),
            torch.nn.Linear(base.cls_score.in_features, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
        )
        torch.nn.init.zeros_(self.adapter[-1].weight)
        torch.nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, features):
        if features.ndim == 4:
            features = features.flatten(1)
        logits, boxes = self.base(features)
        correction = 2 * torch.tanh(self.adapter(features).squeeze(1) / 2)
        return torch.stack([logits[:, 0], logits[:, 1] + correction], dim=1), boxes


def install_roi_adapter(model):
    if isinstance(model.roi_heads.box_predictor, ResidualSignalPredictor):
        raise ValueError("Adapter already installed")
    model.requires_grad_(False)
    model.roi_heads.box_predictor = ResidualSignalPredictor(model.roi_heads.box_predictor)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 65665
    return model


def make_signal_detector(coco_state=None, *, roi_adapter=False):
    """Adapt COCO class10 to binary housing detection; freeze expensive features."""
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91)
    if coco_state is not None:
        model.load_state_dict(coco_state, strict=True)
    old = model.roi_heads
    predictor = FastRCNNPredictor(old.box_predictor.cls_score.in_features, 2)
    with torch.no_grad():
        for name in ("weight", "bias"):
            getattr(predictor.cls_score, name).copy_(
                getattr(old.box_predictor.cls_score, name)[[0, 10]]
            )
            box = getattr(old.box_predictor.bbox_pred, name).reshape(91, 4, -1)[[0, 10]]
            getattr(predictor.bbox_pred, name).copy_(
                box.reshape_as(getattr(predictor.bbox_pred, name))
            )
    model.roi_heads = IgnoreRoIHeads(
        old.box_roi_pool,
        old.box_head,
        predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=64,
        positive_fraction=0.25,
        bbox_reg_weights=old.box_coder.weights,
        score_thresh=0.5,
        nms_thresh=0.5,
        detections_per_img=100,
    )
    rpn = model.rpn
    # Square-root area sizes in resized-image pixels; ratio means height/width.
    anchors = AnchorGenerator(((8,), (16,), (32,), (64,), (128,)), ((1.0, 2.0, 4.0),) * 5)
    model.rpn = IgnoreRPN(
        anchors,
        rpn.head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n=rpn._pre_nms_top_n,
        post_nms_top_n=rpn._post_nms_top_n,
        nms_thresh=rpn.nms_thresh,
        score_thresh=rpn.score_thresh,
    )
    original = model.transform
    model.transform = IgnoreTransform(
        original.min_size,
        original.max_size,
        original.image_mean,
        original.image_std,
        size_divisible=original.size_divisible,
        fixed_size=original.fixed_size,
    )
    model.backbone.requires_grad_(False)
    model.roi_heads.box_head.requires_grad_(False)
    if roi_adapter:
        install_roi_adapter(model)
    return model


def detector_training(model):
    """Train detector samplers/heads while retaining frozen features and BN buffers."""
    model.train()
    model.backbone.eval()
    model.roi_heads.box_head.eval()
