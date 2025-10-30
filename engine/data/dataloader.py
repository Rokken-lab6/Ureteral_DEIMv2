"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register
torchvision.disable_beta_transforms_warning()
from copy import deepcopy
from PIL import Image, ImageDraw
import os
from collections import defaultdict, deque
from pathlib import Path


__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


class SequenceAwareBatchSampler(data.BatchSampler):
    """Batch sampler that limits per-sequence usage and background frequency."""

    def __init__(
        self,
        sampler,
        batch_size,
        drop_last,
        *,
        sequence_lookup=None,
        is_background_lookup=None,
        single_image_per_sequence=False,
        background_image_per_batch=None,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.single_image_per_sequence = single_image_per_sequence
        self.background_image_per_batch = background_image_per_batch
        self.sequence_lookup = sequence_lookup or {}
        self.is_background_lookup = is_background_lookup or {}
        self._planned_order = None
        self._planned_length = None
        self._plan_consumed = False
        self._last_yielded = None
        self._exhausted_early = False

    def _get_sequence_id(self, index):
        return self.sequence_lookup.get(index, index)

    def _is_background(self, index):
        return self.is_background_lookup.get(index, False)

    def refresh_plan(self):
        self._planned_order = None
        self._planned_length = None
        self._plan_consumed = False

    def _ensure_plan(self):
        if self._planned_order is not None:
            return
        order = list(self.sampler)
        self._planned_length, exhausted = self._simulate_length(order)
        self._planned_order = order
        self._exhausted_early = exhausted
        self._plan_consumed = False

    def _simulate_length(self, order):
        if not order:
            return 0, False

        indices_queue = deque(order)
        total_batches = 0
        exhausted = False
        required_background = self.background_image_per_batch
        if required_background is not None and required_background > self.batch_size:
            return 0, True

        while indices_queue:
            batch = []
            used_sequences = set()
            background_count = 0
            consecutive_failures = 0

            while indices_queue and len(batch) < self.batch_size:
                index = indices_queue.popleft()
                seq_id = self._get_sequence_id(index)
                is_background = self._is_background(index)
                remaining_slots = self.batch_size - len(batch)
                remaining_background_needed = (
                    required_background - background_count if required_background is not None else 0
                )
                must_preserve_slot_for_background = (
                    required_background is not None and not is_background and remaining_background_needed >= remaining_slots
                )

                violates_sequence = self.single_image_per_sequence and seq_id in used_sequences
                violates_background = (
                    required_background is not None
                    and (
                        (is_background and background_count >= required_background)
                        or must_preserve_slot_for_background
                    )
                )

                if violates_sequence or violates_background:
                    indices_queue.append(index)
                    consecutive_failures += 1
                    if consecutive_failures >= len(indices_queue):
                        break
                    continue

                batch.append(index)
                if self.single_image_per_sequence:
                    used_sequences.add(seq_id)
                if required_background is not None and is_background:
                    background_count += 1
                consecutive_failures = 0

            if len(batch) == self.batch_size and (
                required_background is None or background_count == required_background
            ):
                total_batches += 1
            else:
                exhausted = True
                break

        return total_batches, exhausted

    def __iter__(self):
        if not self.single_image_per_sequence and self.background_image_per_batch is None:
            yield from super().__iter__()
            return

        self._ensure_plan()
        indices_queue = deque(self._planned_order)
        yielded = 0
        exhausted = False
        required_background = self.background_image_per_batch
        if required_background is not None and required_background > self.batch_size:
            self._last_yielded = 0
            self._exhausted_early = True
            self._plan_consumed = True
            self._planned_order = None
            self._planned_length = None
            return

        while indices_queue:
            batch = []
            used_sequences = set()
            background_count = 0
            consecutive_failures = 0

            while indices_queue and len(batch) < self.batch_size:
                index = indices_queue.popleft()
                seq_id = self._get_sequence_id(index)
                is_background = self._is_background(index)
                remaining_slots = self.batch_size - len(batch)
                remaining_background_needed = (
                    required_background - background_count if required_background is not None else 0
                )
                must_preserve_slot_for_background = (
                    required_background is not None and not is_background and remaining_background_needed >= remaining_slots
                )

                violates_sequence = self.single_image_per_sequence and seq_id in used_sequences
                violates_background = (
                    required_background is not None
                    and (
                        (is_background and background_count >= required_background)
                        or must_preserve_slot_for_background
                    )
                )

                if violates_sequence or violates_background:
                    indices_queue.append(index)
                    consecutive_failures += 1
                    if consecutive_failures >= len(indices_queue):
                        break
                    continue

                batch.append(index)
                if self.single_image_per_sequence:
                    used_sequences.add(seq_id)
                if required_background is not None and is_background:
                    background_count += 1
                consecutive_failures = 0

            if len(batch) == self.batch_size and (
                required_background is None or background_count == required_background
            ):
                yielded += 1
                yield batch
            else:
                exhausted = True
                break

        self._last_yielded = yielded
        self._exhausted_early = exhausted
        self._plan_consumed = True
        self._planned_order = None
        self._planned_length = None

    def __len__(self):
        if not self.single_image_per_sequence and self.background_image_per_batch is None:
            return super().__len__()
        if self._planned_order is None:
            if self._plan_consumed and self._last_yielded is not None:
                return self._last_yielded
            self._ensure_plan()
        return self._planned_length or 0

    @property
    def last_yielded(self):
        return self._last_yielded

    @property
    def exhausted_early(self):
        return self._exhausted_early


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory_device='',
        single_image_per_sequence=False,
        background_image_per_batch=None,
        training_steps=None,
        inference_batch_size=None,
    ) -> None:
        self.single_image_per_sequence = bool(single_image_per_sequence)
        self.background_image_per_batch = (
            int(background_image_per_batch) if background_image_per_batch is not None else None
        )
        self._training_steps = training_steps if training_steps is None else int(training_steps)
        self.inference_batch_size = (
            int(inference_batch_size) if inference_batch_size is not None else None
        )
        if self.inference_batch_size is not None and self.inference_batch_size <= 0:
            raise ValueError('inference_batch_size must be positive if provided')
        if self.background_image_per_batch is not None:
            self.background_image_per_batch = max(0, self.background_image_per_batch)
        if self._training_steps is not None and self._training_steps <= 0:
            self._training_steps = None

        use_sequence_sampler = (
            (self.single_image_per_sequence or self.background_image_per_batch is not None)
            and batch_sampler is None
        )

        sequence_lookup = {}
        background_lookup = {}

        self._last_epoch_steps = None
        self._last_epoch_exhausted = False

        init_kwargs = {
            'dataset': dataset,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            'pin_memory': pin_memory,
            'timeout': timeout,
            'worker_init_fn': worker_init_fn,
            'multiprocessing_context': multiprocessing_context,
            'generator': generator,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': persistent_workers,
            'pin_memory_device': pin_memory_device,
        }

        if use_sequence_sampler:
            sequence_lookup, background_lookup = self._build_dataset_metadata(
                dataset,
                self.background_image_per_batch,
            )

            if sampler is None:
                sampler = data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)

            init_kwargs['batch_sampler'] = SequenceAwareBatchSampler(
                sampler,
                batch_size,
                drop_last,
                sequence_lookup=sequence_lookup,
                is_background_lookup=background_lookup,
                single_image_per_sequence=self.single_image_per_sequence,
                background_image_per_batch=self.background_image_per_batch,
            )
        else:
            init_kwargs.update(
                {
                    'batch_size': batch_size,
                    'shuffle': shuffle,
                    'sampler': sampler,
                    'batch_sampler': batch_sampler,
                    'drop_last': drop_last,
                }
            )

        super().__init__(**init_kwargs)

        self._sequence_lookup = sequence_lookup
        self._is_background_lookup = background_lookup

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle

    def __iter__(self):
        base_iterator = super().__iter__()
        counted = 0

        if self._training_steps is not None and self._training_steps > 0:
            for batch in base_iterator:
                if counted >= self._training_steps:
                    break
                counted += 1
                yield batch
        else:
            for batch in base_iterator:
                counted += 1
                yield batch

        self._last_epoch_steps = counted

        if isinstance(getattr(self, 'batch_sampler', None), SequenceAwareBatchSampler):
            sampler = self.batch_sampler
            self._last_epoch_exhausted = sampler.exhausted_early and (
                self._training_steps is None or counted < self._training_steps
            )
            if sampler.last_yielded is not None and self._training_steps is None:
                self._last_epoch_steps = sampler.last_yielded
            if self._last_epoch_steps is not None:
                status = ' (constraints reached)' if self._last_epoch_exhausted else ''
                print(
                    f"[DataLoader] Effective batches this epoch: {self._last_epoch_steps}{status}"
                )

    def __len__(self):
        base_len = super().__len__()
        if self._training_steps is None or self._training_steps <= 0:
            if self._last_epoch_steps is not None:
                return self._last_epoch_steps
            return base_len
        return min(self._training_steps, base_len)

    @property
    def last_epoch_steps(self):
        return self._last_epoch_steps

    @staticmethod
    def _build_dataset_metadata(dataset, background_limit):
        sequence_lookup = {}
        background_lookup = {}

        try:
            if hasattr(dataset, 'coco') and hasattr(dataset, 'ids'):
                coco = dataset.coco
                for index, img_id in enumerate(dataset.ids):
                    info = coco.loadImgs(img_id)[0]
                    file_name = info.get('file_name', str(img_id))
                    sequence_lookup[index] = DataLoader._extract_sequence_id(file_name)
                    if background_limit is not None:
                        ann_ids = coco.getAnnIds(imgIds=[img_id])
                        background_lookup[index] = len(ann_ids) == 0
            elif hasattr(dataset, 'imgs'):
                entries = dataset.imgs
                if isinstance(entries, dict):
                    iterable = entries.values()
                else:
                    iterable = entries
                for index, item in enumerate(iterable):
                    if isinstance(item, (list, tuple)):
                        path = item[0]
                        target = item[1] if len(item) > 1 else None
                    else:
                        path, target = item, None
                    sequence_lookup[index] = DataLoader._extract_sequence_id(path)
                    if background_limit is not None:
                        background_lookup[index] = DataLoader._is_empty_target(target)
        except Exception:
            sequence_lookup = {}
            background_lookup = {}

        return sequence_lookup, background_lookup

    @staticmethod
    def _extract_sequence_id(file_name):
        path = Path(file_name)
        if path.parent != Path('') and path.parent != Path('.'):
            return path.parent.name
        return path.stem

    @staticmethod
    def _is_empty_target(target):
        if target is None:
            return True
        if isinstance(target, (list, tuple)):
            return len(target) == 0
        if isinstance(target, dict):
            boxes = target.get('boxes') or target.get('annotations')
            return not boxes
        return False


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register() 
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        stop_epoch=None, 
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        mixup_prob=0.0,
        mixup_epochs=[0, 0],
        copyblend_prob=0.0,
        copyblend_epochs=[0, 0],
        copyblend_type='blend',
        conflict_with_mixup=False,
        area_threshold=100,
        num_objects=3,
        with_expand=False,
        expand_ratios=[0.1, 0.25],
        random_num_objects=False,
        data_vis=False,
        vis_save='./vis_dataset/'
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        self.mixup_prob, self.mixup_epochs = mixup_prob, mixup_epochs

        self.copyblend_prob, self.copyblend_epochs, self.copyblend_type = copyblend_prob, copyblend_epochs, copyblend_type
        self.area_threshold, self.num_objects = area_threshold, num_objects
        self.data_vis, self.vis_save = data_vis, vis_save
        self.with_expand, self.expand_ratios, self.random_num_objects = with_expand, expand_ratios, random_num_objects
        self.conflict_with_mixup = conflict_with_mixup  # 是否冲突

        if self.mixup_prob > 0 or self.copyblend_prob > 0:
            if os.path.isdir(self.vis_save):
                for file in os.listdir(self.vis_save):
                    os.remove('{}/{}'.format(self.vis_save, file))
            os.makedirs(self.vis_save, exist_ok=True) if self.data_vis else None

            if self.mixup_prob > 0:
                print("     ### Using MixUp with Prob@{} in {} epochs ### ".format(mixup_prob, mixup_epochs))
            if self.copyblend_prob > 0:
                print("     ### Using CopyBlend-{} with Prob@{} in {} epochs ### ".format(copyblend_type, copyblend_prob, copyblend_epochs))
                print(f'     ### CopyBlend -- area threshold@{area_threshold} and num of object@{num_objects} ###     ')
                if self.with_expand:
                    print(f'     ### CopyBlend -- expand@{expand_ratios} ###     ')
                if self.random_num_objects:
                    print(f'     ### CopyBlend -- random num of objects@{[1, self.num_objects]} ###     ')

        if stop_epoch is not None:
            print("     ### Multi-scale Training until {} epochs ### ".format(self.stop_epoch))
            print("     ### Multi-scales@ {} ###        ".format(self.scales))
        self.print_info_flag = True
        self.print_copyblend_flag = True
        # self.interpolation = interpolation

    def apply_mixup(self, images, targets):
        """
        Applies Mixup augmentation to the batch if conditions are met.

        Args:
            images (torch.Tensor): Batch of images.
            targets (list[dict]): List of target dictionaries corresponding to images.

        Returns:
            tuple: Updated images and targets
        """
        # Log when Mixup is permanently disabled
        if self.epoch == self.mixup_epochs[-1] and self.print_info_flag:
            print(f"     ### Attention --- Mixup is closed after epoch@ {self.epoch} ###")
            self.print_info_flag = False

        MixUp_flag, CopyBlend_flag = False, False
        beta = round(random.uniform(0.45, 0.55), 6)
        # Apply Mixup if within specified epoch range and probability threshold
        if random.random() < self.mixup_prob and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]:
            # Generate mixup ratio
            beta = round(random.uniform(0.45, 0.55), 6)
            MixUp_flag = True

            # Mix images
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

            # Prepare targets for Mixup
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = deepcopy(targets)

            for i in range(len(targets)):
                # Combine boxes, labels, and areas from original and shifted targets
                updated_targets[i]['boxes'] = torch.cat([targets[i]['boxes'], shifted_targets[i]['boxes']], dim=0)
                updated_targets[i]['labels'] = torch.cat([targets[i]['labels'], shifted_targets[i]['labels']], dim=0)
                updated_targets[i]['area'] = torch.cat([targets[i]['area'], shifted_targets[i]['area']], dim=0)

                # Add mixup ratio to targets
                updated_targets[i]['mixup'] = torch.tensor(
                    [beta] * len(targets[i]['labels']) + [1.0 - beta] * len(shifted_targets[i]['labels']), 
                    dtype=torch.float32
                    )
            targets = updated_targets

        elif (self.copyblend_epochs[0] <= self.epoch < self.copyblend_epochs[-1] and random.random() < self.copyblend_prob):
            if self.epoch == self.copyblend_epochs[-1] and self.print_copyblend_flag:
                print(f"     ### Attention --- CopyBlend closed after epoch@ {self.epoch} ###")
                self.print_copyblend_flag = False

            CopyBlend_flag = True
            objects_pool = defaultdict(list)
            img_height, img_width = images[0].shape[-2:]

            # get all valid objects in batch
            for i in range(len(images)):
                source_boxes = targets[i]['boxes']
                source_labels = targets[i]['labels']
                source_areas = targets[i]['area']
                
                # filter valid objects
                valid_objects = [idx for idx in range(len(source_boxes)) if source_areas[idx] >= self.area_threshold]
                for idx in valid_objects:
                    objects_pool['boxes'].append(source_boxes[idx])
                    objects_pool['labels'].append(source_labels[idx])
                    objects_pool['areas'].append(source_areas[idx])
                    objects_pool['image_idx'].append(i)
                    objects_pool['image_height'].append(img_height)
                    objects_pool['image_width'].append(img_width)
            
            # check if objects_pool is empty
            if len(objects_pool['boxes']) == 0:
                return images, targets
            
            # convert list to tensor for convenient operation
            for key in ['boxes', 'labels', 'areas']:
                objects_pool[key] = torch.stack(objects_pool[key]) if objects_pool[key] else torch.tensor([])
                
            # apply CopyBlend
            batch_size = len(images)
            updated_images = images.clone()
            updated_targets = deepcopy(targets)

            for i in range(batch_size):
                # randomly decide the number of objects to blend
                if self.random_num_objects:
                    num_objects = random.randint(1, min(self.num_objects, len(objects_pool['boxes'])))
                else:
                    num_objects = min(self.num_objects, len(objects_pool['boxes']))
                
                # randomly select objects to blend
                selected_indices = random.sample(range(len(objects_pool['boxes'])), num_objects)
                
                blend_boxes = []
                blend_labels = []
                blend_areas = []
                blend_mixup_ratios = []

                for idx in selected_indices:
                    # get source object information
                    box = objects_pool['boxes'][idx]
                    label = objects_pool['labels'][idx]
                    area = objects_pool['areas'][idx]
                    source_idx = objects_pool['image_idx'][idx]
                    source_height = objects_pool['image_height'][idx]
                    source_width = objects_pool['image_width'][idx]
                    
                    # calculate source object size and position
                    cx, cy, w, h = box
                    x1_src, y1_src = int((cx - w / 2) * source_width), int((cy - h / 2) * source_height)
                    x2_src, y2_src = int((cx + w / 2) * source_width), int((cy + h / 2) * source_height)

                    # check if source object is out of bound
                    x1_src, y1_src = max(x1_src, 0), max(y1_src, 0)
                    x2_src, y2_src = min(x2_src, img_width), min(y2_src, img_height)
                    new_w_px, new_h_px = x2_src - x1_src, y2_src - y1_src
                    # check if source object is valid
                    if new_w_px <= 0 or new_h_px <= 0:
                        continue

                    # randomly determine blend position
                    x1 = random.randint(0, img_width - new_w_px) if new_w_px < img_width else 0
                    y1 = random.randint(0, img_height - new_h_px) if new_h_px < img_height else 0
                    # after the above limit, [x2, y2] will not be out of bound, so no need to check
                    x2, y2 = x1 + new_w_px, y1 + new_h_px
                    
                    # calculate new normalized coordinates
                    new_cx, new_cy = (x1 + new_w_px / 2) / img_width, (y1 + new_h_px / 2) / img_height
                    new_w, new_h = new_w_px / img_width, new_h_px / img_height

                    # add to blend list - use original unexpanded box
                    blend_boxes.append(torch.tensor([new_cx, new_cy, new_w, new_h]))
                    blend_labels.append(label)
                    blend_areas.append(area)
                    # mixup ratio
                    blend_mixup_ratios.append(1.0 - beta)

                    # handle expanded area
                    if self.with_expand:
                        alpha = round(random.uniform(self.expand_ratios[0], self.expand_ratios[1]), 6)
                        expand_w, expand_h = int(new_w_px * alpha), int(new_h_px * alpha)
                        # check if out of bound: get the best offset in GT image
                        x1_expand, y1_expand = x1_src - max(x1_src - expand_w, 0), y1_src - max(y1_src - expand_h, 0)
                        x2_expand, y2_expand = min(x2_src + expand_w, img_width) - x2_src, min(y2_src + expand_h, img_height) - y2_src
                        # check if out of bound: whether the expanded area is out of bound in blend image
                        new_x1_expand, new_y1_expand = x1 - max(x1 - x1_expand, 0), y1 - max(y1 - y1_expand, 0)
                        new_x2_expand, new_y2_expand = min(x2 + x2_expand, img_width) - x2, min(y2 + y2_expand, img_height) - y2
                        # update
                        x1_src, y1_src, x2_src, y2_src = x1_src - new_x1_expand, y1_src - new_y1_expand, x2_src + new_x2_expand, y2_src + new_y2_expand
                        x1, y1, x2, y2 = x1 - new_x1_expand, y1 - new_y1_expand, x2 + new_x2_expand, y2 + new_y2_expand

                    # blend original area first
                    copy_patch_orig = images[source_idx, :, y1_src:y2_src, x1_src:x2_src]
                    if self.copyblend_type == 'blend':
                        blended_patch = updated_images[i, :, y1:y2, x1:x2] * beta + copy_patch_orig * (1 - beta)
                        updated_images[i, :, y1:y2, x1:x2] = blended_patch
                    else:
                        updated_images[i, :, y1:y2, x1:x2] = copy_patch_orig
                    
                # add blended objects to targets
                if len(blend_boxes) > 0:
                    blend_boxes = torch.stack(blend_boxes)
                    blend_labels = torch.stack(blend_labels)
                    blend_areas = torch.stack(blend_areas)
                    
                    # add mixup ratio
                    updated_targets[i]['mixup'] = torch.tensor(
                        [1.0] * len(updated_targets[i]['boxes']) + blend_mixup_ratios, 
                        dtype=torch.float32
                    )
                    # update targets
                    updated_targets[i]['boxes'] = torch.cat([updated_targets[i]['boxes'], blend_boxes])
                    updated_targets[i]['labels'] = torch.cat([updated_targets[i]['labels'], blend_labels])
                    updated_targets[i]['area'] = torch.cat([updated_targets[i]['area'], blend_areas])

            images, targets = updated_images, updated_targets

            if self.data_vis and CopyBlend_flag:
                for i in range(len(updated_targets)):
                    image_tensor = images[i]
                    if image_tensor.min() < 0:  # use normalization
                        image_tensor = image_tensor * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) \
                            + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    image_tensor_uint8 = (image_tensor * 255).type(torch.uint8)
                    image_numpy = image_tensor_uint8.numpy().transpose((1, 2, 0))
                    pilImage = Image.fromarray(image_numpy)
                    draw = ImageDraw.Draw(pilImage)
                    print('mix_vis:', i, 'boxes.len=', len(updated_targets[i]['boxes']))
                    for box in updated_targets[i]['boxes']:
                        draw.rectangle([int(box[0]*640 - (box[2]*640)/2), int(box[1]*640 - (box[3]*640)/2), 
                                        int(box[0]*640 + (box[2]*640)/2), int(box[1]*640 + (box[3]*640)/2)], outline=(255,255,0))
                    pilImage.save(self.vis_save + str(i) + "_"+ str(len(updated_targets[i]['boxes'])) +'_out.jpg')

        return images, targets

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        # Mixup
        images, targets = self.apply_mixup(images, targets)

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        return images, targets
