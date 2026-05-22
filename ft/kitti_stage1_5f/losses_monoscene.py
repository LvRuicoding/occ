"""MonoScene-style SSC loss for the kitti_stage1_5f monoscene experiment.

Composed of (weights all 1, matching MonoScene's KITTI config):
    - CE_ssc_loss          (cross-entropy with class weights, ignore=255)
    - sem_scal_loss        (semantic precision/recall/specificity BCE)
    - geo_scal_loss        (geometric scale-aware BCE: empty vs non-empty)
    - relation_ce_super    (BCE-with-logits on the CP_mega_matrix relations)

The frustum-proportion (fp) loss is intentionally omitted -- the lifted
dataset doesn't carry the per-frustum masks needed for it.

Implementations of the four terms are taken verbatim (modulo imports)
from MonoScene's monoscene/loss/{ssc_loss,CRP_loss}.py.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ft.semantickitti_ft.losses import class_weights_from_frequencies


def CE_ssc_loss(pred: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    return criterion(pred, target.long())


def geo_scal_loss(pred: torch.Tensor, ssc_target: torch.Tensor) -> torch.Tensor:
    # F.binary_cross_entropy is autocast-unsafe; run this loss in fp32.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        pred = F.softmax(pred.float(), dim=1)

        empty_probs = pred[:, 0, :, :, :]
        nonempty_probs = 1 - empty_probs

        mask = ssc_target != 255
        nonempty_target = ssc_target != 0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )


def sem_scal_loss(pred: torch.Tensor, ssc_target: torch.Tensor) -> torch.Tensor:
    # F.binary_cross_entropy is autocast-unsafe; run this loss in fp32.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        pred = F.softmax(pred.float(), dim=1)
        loss = 0
        count = 0
        mask = ssc_target != 255
        n_classes = pred.shape[1]
        for i in range(0, n_classes):
            p = pred[:, i, :, :, :]

            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count


def compute_super_CP_multilabel_loss(
    pred_logits: torch.Tensor, CP_mega_matrices: torch.Tensor
) -> torch.Tensor:
    logits = []
    labels = []
    bs, n_relations, _, _ = pred_logits.shape
    for i in range(bs):
        pred_logit = pred_logits[i, :, :, :].permute(0, 2, 1)  # n_rel, N, N_super
        CP_mega_matrix = CP_mega_matrices[i]  # n_rel, N, N_super
        logits.append(pred_logit.reshape(n_relations, -1))
        labels.append(CP_mega_matrix.reshape(n_relations, -1))

    logits = torch.cat(logits, dim=1).T  # M, n_rel
    labels = torch.cat(labels, dim=1).T

    cnt_neg = (labels == 0).sum(0)
    cnt_pos = labels.sum(0)
    pos_weight = cnt_neg / cnt_pos
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion(logits, labels.float())


class MonoSceneSSCLoss(nn.Module):
    """MonoScene combined loss (CE + sem_scal + geo_scal + relation_ce)."""

    def __init__(
        self,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        sem_scal_weight: float = 1.0,
        geo_scal_weight: float = 1.0,
        relation_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.sem_scal_weight = float(sem_scal_weight)
        self.geo_scal_weight = float(geo_scal_weight)
        self.relation_weight = float(relation_weight)
        if class_weights is None:
            class_weights = class_weights_from_frequencies()
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        target: torch.Tensor,
        cp_mega_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ssc_logit = model_out["ssc_logit"]

        cw = self.class_weights.to(ssc_logit.dtype)
        loss_ce = CE_ssc_loss(ssc_logit, target, cw)
        loss_sem = sem_scal_loss(ssc_logit, target)
        loss_geo = geo_scal_loss(ssc_logit, target)

        if "P_logits" not in model_out:
            raise RuntimeError(
                "MonoSceneSSCLoss expected model_out to contain 'P_logits' "
                "(context_prior must be enabled)."
            )
        loss_rel = compute_super_CP_multilabel_loss(
            model_out["P_logits"], cp_mega_matrix
        )

        loss = (
            self.ce_weight * loss_ce
            + self.sem_scal_weight * loss_sem
            + self.geo_scal_weight * loss_geo
            + self.relation_weight * loss_rel
        )
        details = dict(
            ce=float(loss_ce.detach()),
            sem_scal=float(loss_sem.detach()),
            geo_scal=float(loss_geo.detach()),
            relation_ce=float(loss_rel.detach()),
        )
        return loss, details
