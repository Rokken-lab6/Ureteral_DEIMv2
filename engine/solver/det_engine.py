"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch_size = len(targets) if isinstance(targets, (list, tuple)) else samples.shape[0]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        inference_batch_size = getattr(data_loader, 'inference_batch_size', None)
        if inference_batch_size is None or inference_batch_size <= 0:
            micro_slices = [(0, batch_size)]
        else:
            micro = max(1, min(int(inference_batch_size), batch_size))
            micro_slices = [(start, min(start + micro, batch_size)) for start in range(0, batch_size, micro)]

        total_chunks = len(micro_slices)
        optimizer.zero_grad(set_to_none=True)

        aggregated_loss_dict = {}

        for micro_idx, (start, end) in enumerate(micro_slices):
            chunk_size = end - start
            chunk_weight = chunk_size / batch_size if batch_size > 0 else 1.0

            samples_chunk = samples[start:end].to(device, non_blocking=True)
            targets_slice = targets[start:end]
            targets_chunk = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets_slice]

            chunk_metas = metas.copy()
            chunk_metas.update({
                'micro_batch_index': micro_idx,
                'micro_batch_count': total_chunks,
                'virtual_batch_size': batch_size,
            })

            if scaler is not None:
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    outputs = model(samples_chunk, targets=targets_chunk)

                if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                    print(outputs['pred_boxes'])
                    state = model.state_dict()
                    new_state = {}
                    for key, value in model.state_dict().items():
                        # Replace 'module' with 'model' in each key
                        new_key = key.replace('module.', '')
                        # Add the updated key-value pair to the state dictionary
                        state[new_key] = value
                    new_state['model'] = state
                    dist_utils.save_on_master(new_state, "./NaN.pth")

                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets_chunk, **chunk_metas)
            else:
                outputs = model(samples_chunk, targets=targets_chunk)
                loss_dict = criterion(outputs, targets_chunk, **chunk_metas)

            loss_dict = {k: v * chunk_weight for k, v in loss_dict.items()}
            loss = sum(loss_dict.values())

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            for k, v in loss_dict.items():
                if k in aggregated_loss_dict:
                    aggregated_loss_dict[k] = aggregated_loss_dict[k] + v.detach()
                else:
                    aggregated_loss_dict[k] = v.detach()

        if scaler is not None:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        loss_dict = aggregated_loss_dict

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    distance_threshold: float,
):
    model.eval()
    criterion.eval()
    if coco_evaluator is not None:
        coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    threshold_steps = list(range(0, 101))
    match_counts = {step: {'tp': 0.0, 'fp': 0.0, 'fn': 0.0} for step in threshold_steps}

    distance_threshold = float(distance_threshold) if distance_threshold is not None else 0.0
    distance_threshold = max(distance_threshold, 0.0)

    use_focal_loss = getattr(postprocessor, 'use_focal_loss', False)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        img_h, img_w = samples.shape[-2:]
        scale = pred_boxes.new_tensor([img_w, img_h])

        for idx, target in enumerate(targets):
            logits = pred_logits[idx]
            boxes = pred_boxes[idx]

            if use_focal_loss:
                scores = torch.sigmoid(logits)
                scores_flat = scores.reshape(-1)
                top_score, top_index = torch.max(scores_flat, dim=0)
                num_classes = scores.shape[-1]
                query_idx = int(top_index.item() // max(num_classes, 1))
            else:
                probs = torch.softmax(logits, dim=-1)
                if probs.shape[-1] > 1:
                    probs_without_bg = probs[..., :-1]
                else:
                    probs_without_bg = probs
                top_scores_per_query, _ = torch.max(probs_without_bg, dim=-1)
                top_score, query_idx = torch.max(top_scores_per_query, dim=0)
                query_idx = int(query_idx.item())

            if boxes.shape[0] == 0:
                continue

            query_idx = max(min(query_idx, boxes.shape[0] - 1), 0)
            top_score = float(torch.clamp(top_score, min=0.0, max=1.0).item())

            pred_center = boxes[query_idx, :2]
            finite_mask = torch.isfinite(pred_center)
            pred_center = torch.where(finite_mask, pred_center, torch.zeros_like(pred_center))
            pred_center = pred_center.clamp(0.0, 1.0)
            pred_center_px = pred_center * scale

            gt_boxes = target.get('boxes', None)
            has_gt = gt_boxes is not None and gt_boxes.numel() > 0
            num_gt = int(gt_boxes.shape[0]) if has_gt else 0

            min_distance = None
            if has_gt:
                gt_boxes_tensor = gt_boxes.to(dtype=pred_center.dtype)
                gt_centers_px = torch.stack(
                    ((gt_boxes_tensor[:, 0] + gt_boxes_tensor[:, 2]) * 0.5,
                     (gt_boxes_tensor[:, 1] + gt_boxes_tensor[:, 3]) * 0.5),
                    dim=-1
                )
                deltas = gt_centers_px - pred_center_px.unsqueeze(0)
                distances = torch.linalg.norm(deltas, dim=-1)
                if distances.numel() > 0:
                    min_distance = float(distances.min().item())

            for step in threshold_steps:
                thr_value = step / 100.0
                if top_score >= thr_value:
                    if has_gt and min_distance is not None and min_distance < distance_threshold:
                        match_counts[step]['tp'] += 1
                        if num_gt > 1:
                            match_counts[step]['fn'] += num_gt - 1
                    else:
                        match_counts[step]['fp'] += 1
                        if has_gt:
                            match_counts[step]['fn'] += num_gt
                else:
                    if has_gt:
                        match_counts[step]['fn'] += num_gt

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    reduce_input = {}
    for step in threshold_steps:
        for key in ('tp', 'fp', 'fn'):
            reduce_input[f'{key}_{step}'] = torch.tensor(match_counts[step][key], device=device)

    reduced_counts = dist_utils.reduce_dict(reduce_input, avg=False)
    for step in threshold_steps:
        for key in ('tp', 'fp', 'fn'):
            match_counts[step][key] = float(reduced_counts[f'{key}_{step}'].item())

    f1_per_threshold = {}
    best_f1 = 0.0
    best_threshold = 0.0

    for step in threshold_steps:
        thr_value = step / 100.0
        thr_key = round(thr_value, 2)
        tp = match_counts[step]['tp']
        fp = match_counts[step]['fp']
        fn = match_counts[step]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_per_threshold[thr_key] = f1

        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-8 and thr_key > best_threshold):
            best_f1 = f1
            best_threshold = thr_key

    stats = {
        'best_f1': best_f1,
        'best_f1_threshold': best_threshold,
        'best_f1_conf_threshold': best_threshold,
        'f1_per_threshold': f1_per_threshold,
    }

    return stats, None
