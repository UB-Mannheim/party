#
# Copyright 2024 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
conformer_ocr.rpred
~~~~~~~~~~~~

Generators for recognition on lines images.
"""
import uuid
import math
import torch
import dataclasses
import logging
from itertools import islice
from multiprocessing import Pool
from collections import defaultdict
from functools import partial
from typing import (TYPE_CHECKING, Dict, Generator, List, Optional, Sequence,
                    Tuple, Union)

import warnings
import torch.nn.functional as F

from kraken.containers import BaselineOCRRecord, BBoxOCRRecord, ocr_record
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.segmentation import extract_polygons
from kraken.lib.util import get_im_str, is_bitonal

if TYPE_CHECKING:
    from PIL import Image

    from kraken.containers import Segmentation
    from conformer_ocr.pred import PytorchRecognitionModel

__all__ = ['rpred']

logger = logging.getLogger(__name__)



def collate_sequences(batch):
    """
    Pads sequences.
    """
    seq_lens = torch.LongTensor([seq.shape[2] for seq in batch])
    max_len = max(seq_lens)
    seqs = torch.stack([F.pad(seq, pad=(0, max_len-seq.shape[2])) for seq in batch])
    return seqs, seq_lens


class LineImageDataset(Dataset):
    def __init__(self,
                 bounds: 'Segmentation',
                 line_ids: Sequence['UUID'],
                 transforms: 'ImageInputTransforms'):
        self.im = Image.open(seg.imagename)
        self.bounds = bounds
        self.ids = line_ids
        self.ts = transforms
        self.pad = transforms.pad

    def __len__(self):
        return len(self.bounds.lines)

    def __getitem__(self, idx):
        line = self.bounds.line[idx]
        _seg = dataclasses.replace(self.bounds, lines=[line])
        im, _ = next(extract_polygons(self.im, seg))
        broken = False
        if 0 in im.size:
            logger.warning(f'Line with zero dimension. Emitting empty record.')
            broken = True
        # try conversion into tensor
        try:
            ts_im = self.ts(im)
        except Exception as e:
            logger.warning(f'Tensor conversion failed with {e}. Emitting empty record.')
            broken = True
        # check if line is non-zero
        if ts_box.max() == ts_box.min():
            logger.warning('Empty line after tensor conversion. Emitting empty record.')
            broken = True

        return {'broken': broken,
                'image': im,
                'line': line,
                'line_id': self.ids[idx],
                'in_scales': im/(ts_box.shape[2]-2*self.pad) if not broken else -1.0,
                'segmentation': self.bounds}


class rpred(object):
    """
    Batched version of kraken.rpred.rpred
    """
    def __init__(self,
                 net: 'PytorchRecognitionModel',
                 bounds: List['Segmentation'],
                 batch_size: int = -1,
                 num_workers: int = 0,
                 bidi_reordering: Union[bool, str] = True) -> Generator[ocr_record, None, None]:
        """
        Batched version of kraken.rpred.rpred

        Args:
            net: A recognition model.
            im: Image to extract text from
            bounds: A Segmentation data class containing either bounding box or
                    baseline type segmentation.
            batch_size: Batch size to use during prediction. If -1 same as
                        during training will be used.
            bidi_reordering: Reorder classes in the ocr_record according to the
                             Unicode bidirectional algorithm for correct
                             display. Set to L|R to override default text
                             direction.

        Yields:
            An ocr_record containing the recognized text, absolute character
            positions, and confidence values for each character.
        """
        if bounds.type != net.seg_type:
            logger.warning(f'Recognizer with segmentation types {net.seg_type} will be '
                           f'applied to segmentation of type {bounds.type}. '
                           f'This will likely result in severely degraded performance')
        if  net.one_channel_mode == '1' and not is_bitonal(im):
            logger.warning('Running binary models on non-binary input image '
                           f'(mode {im.mode}). This will result in severely degraded '
                           'performance')

        if bounds.type == 'baselines':
            self.next_iter = self._recognize_baseline_line
        else:
            self.next_iter = self._recognize_box_line

        self.ts = ImageInputTransforms(batch_size, net.height, net.width, net.channels, (net.pad, 0), False)

        self.im = im
        self.net = net
        self.bidi_reordering = bidi_reordering self.pad = net.pad
        self.bounds = bounds
        self.batch_size = batch_size if batch_size > 0 else net.batch_size
        self.cache = []

        self.len = len(bounds.lines)
        self.line_iter = _batched(bounds.lines, self.batch_size)

    def _recognize_box_line(self, batch):

        logger.debug(f'Forward pass.')
        inputs, ids, broken_ids = batch['inputs'], batch['ids'], batch['broken']
        preds = self.net.predict(inputs)
        BBoxOCRRecord('', (), (), line)

        self.net_scales = [ts_box.shape[2]/pred_len for ts_box, pred_len in zip(inputs, self.net.output_lens)]
        self.in_scales = batch['in_scale']

        records = []
        for pred, net_scale, in_scale, box_size, line, coords in zip(preds,
                                                                     self.net_scales,
                                                                     self.in_scales,
                                                                     box_sizes,
                                                                     _lines,
                                                                     _coords):
            pred_str = ''.join(x[0] for x in pred)
            pos = []
            conf = []
            line.text_direction = self.bounds.text_direction

            for _, start, end, c in pred:
                if self.bounds.text_direction.startswith('horizontal'):
                    xmin = coords[0] + self._scale_val(start, 0, box_size, net_scale, in_scale)
                    xmax = coords[0] + self._scale_val(end, 0, box_size, net_scale, in_scale)
                    pos.append([[xmin, coords[1]], [xmin, coords[3]], [xmax, coords[3]], [xmax, coords[1]]])
                else:
                    ymin = coords[1] + self._scale_val(start, 0, box_size, net_scale, in_scale)
                    ymax = coords[1] + self._scale_val(end, 0, box_size, net_scale, in_scale)
                    pos.append([[coords[0], ymin], [coords[2], ymin], [coords[2], ymax], [coords[0], ymax]])
                conf.append(c)
            records.append(BBoxOCRRecord(pred_str, pos, conf, line))

        # insert empty records for invalid lines 
        for idx, line in inv_idxs:
            records.insert(idx, BBoxOCRRecord('', [], [], line))

        # reorder records
        if self.bidi_reordering:
            logger.debug('BiDi reordering record.')
            return [rec.logical_order(base_dir=self.bidi_reordering if
                                      self.bidi_reordering in ('L', 'R') else None) for rec in records]
        else:
            return records

    def _recognize_baseline_line(self, lines):
        seg = dataclasses.replace(self.bounds, lines=lines)

        # batch inputs
        inputs = []
        inv_idxs = []
        box_sizes = []
        _lines = []
        for idx, (box, coords) in enumerate(extract_polygons(self.im, seg)):
            # check if boxes are non-zero in any dimension
            if 0 in box.size:
                logger.warning(f'{line} with zero dimension. Emitting empty record.')
                inv_idxs.append((idx, coords))
                continue
            # try conversion into tensor
            try:
                ts_box = self.ts(box)
            except Exception as e:
                logger.warning(f'Tensor conversion failed with {e}. Emitting empty record.')
                inv_idxs.append((idx, coords))
                continue
            # check if line is non-zero
            if ts_box.max() == ts_box.min():
                logger.warning('Empty line after tensor conversion. Emitting empty record.')
                inv_idxs.append((idx, coords))
                continue
            inputs.append(ts_box)
            box_sizes.append(box.size[0])
            _lines.append(coords)

        preds = self.net.predict(*collate_sequences(inputs))[0]
        # calculate recognized network locations of characters scale between
        # network output and network input (does not contain invalid boxes).
        self.net_scales = [ts_box.shape[2]/pred_len for ts_box, pred_len in zip(inputs, self.net.output_lens)]
        # scale between network input and original line.
        self.in_scales = [box/(ts_box.shape[2]-2*self.pad) for box in box_sizes]

        records = []
        for pred, net_scale, in_scale, box_size, line in zip(preds,
                                                             self.net_scales,
                                                             self.in_scales,
                                                             box_sizes,
                                                             _lines):
            pred_str = ''.join(x[0] for x in pred)
            pos = []
            conf = []
            for _, start, end, c in pred:
                pos.append([self._scale_val(start, 0, box_size, net_scale, in_scale),
                            self._scale_val(end, 0, box_size, net_scale, in_scale)])
                conf.append(c)

            records.append(BaselineOCRRecord(pred, pos, conf, line))

        # insert empty records for invalid lines 
        for idx, line in inv_idxs:
            records.insert(idx, BaselineOCRRecord('', [], [], line))

        # reorder records
        if self.bidi_reordering:
            logger.debug('BiDi reordering record.')
            return [rec.logical_order(base_dir=self.bidi_reordering if
                                      self.bidi_reordering in ('L', 'R') else None) for rec in records]
        else:
            return records

    def __next__(self):
        if not self.cache:
            self.cache.extend(self.next_iter(next(self.line_iter)))
        return self.cache.pop(0)

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

    def _scale_val(self, val, min_val, max_val, net_scale, in_scale):
        return int(round(min(max(((val*net_scale)-self.pad)*in_scale, min_val), max_val-1)))
