import base64
import logging
import pathlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from scripts.interactdiffusion_infotext import Infotext

from scripts.interactdiffusion_pluggable import PluggableInteractDiffusion
from scripts import interactdiffusion
from scripts.modules.utils import rescale_js, binarize, center_crop, sized_center_fill,\
      sized_center_mask
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import os

from modules import devices, shared, processing, script_callbacks
import modules.scripts as scripts

from collections import OrderedDict

import torch
from torch import nn
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel
import itertools
from natsort import natsorted
import numpy as np
from modules import shared
from scripts.modules.model import GatedSelfAttentionDense, FourierEmbedder

import logging

from modules.processing import StableDiffusionProcessing


class pipeline():

    def __init__(self):

        self.ori_unet = UNetModel 
        model_path = pathlib.Path(__file__).parent.parent / 'models' / 'ext_interactdiff_v1.2.pth'
        interactdiffusion_state_dict = torch.load(str(model_path), map_location='cuda') #may need to change for distributed compute

        return
    

    def get_clip_feature(self, clipembedder, phrase):
        token_ids = clipembedder.wrapped.tokenizer(phrase, max_length=clipembedder.wrapped.max_length,return_length=True, return_overflowing_tokens=False, return_tensors="pt")
        clip_skip = processing.opts.data['CLIP_stop_at_last_layers']
        outputs = clipembedder.wrapped.transformer(input_ids=token_ids.input_ids.to(device=shared.device), output_hidden_states=-clip_skip)
        if clip_skip > 1:
            z = outputs.hidden_states[-clip_skip]
            z = clipembedder.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        feature = z[[0], [-2],:]

        return feature
    
    def process(self, p: StableDiffusionProcessing, boxes,grounding_texts,enabled,stage_one,stage_two,strength,trained_UNet):#need to refine the way we pass aurguments:
        
        
        self.plug_ID = PluggableInteractDiffusion(trained_UNet,self.interactdiffusion_state_dict)
        grounding_texts = [y for x in grounding_texts.split(';') for y in x.strip().split("=") if y != ""]
        subject_phrases = grounding_texts[0::3]
        action_phrases = grounding_texts[1::3]
        object_phrases = grounding_texts[2::3]

        subject_boxes = boxes[0::2] #need to change to work with boxes type which is undefined as of now.
        object_boxes = boxes[1::2]

        assert 3 * len(boxes) == 2 * len(grounding_texts)

        subject_boxes = (np.asarray(subject_boxes) / 512).tolist()
        object_boxes = (np.asarray(object_boxes) / 512).tolist()
        grounding_instruction = {i: {
            'subject_phrases': s_p,
            'object_phrases': o_p,
            'action_phrases': a_p,
            'subject_boxes': s_b,
            'object_boxes': o_b
        } for i, (s_p, a_p, o_p, s_b, o_b) in
            enumerate(zip(subject_phrases, action_phrases, object_phrases,
                        subject_boxes, object_boxes))}
        
        subject_phrases, object_phrases, action_phrases = [], [], []
        subject_boxes, object_boxes = [], []
        for i, v in grounding_instruction.items():
            # print(f"v: {v}")
            subject_phrases.append(v['subject_phrases'])
            object_phrases.append(v['object_phrases'])
            action_phrases.append(v['action_phrases'])
            subject_boxes.append(v['subject_boxes'])
            object_boxes.append(v['object_boxes'])

        has_text_mask = 1
        has_image_mask = 0
        # get last sep token embedding
        max_objs = 30

        subject_boxes_tensor = torch.zeros(max_objs, 4)
        object_boxes_tensor = torch.zeros(max_objs, 4)
        masks = torch.zeros(max_objs)
        subject_text_embeddings = torch.zeros(max_objs, 768)
        object_text_embeddings = torch.zeros(max_objs, 768)
        action_text_embeddings = torch.zeros(max_objs, 768)

        subject_text_features = []
        object_text_features = []
        action_text_features = []

        clipembedder = shared.sd_model.cond_stage_model

        for subject_phrase, object_phrase, action_phrase \
                in zip(subject_phrases, object_phrases, action_phrases):
            subject_text_features.append(self.get_clip_feature(clipembedder, subject_phrase))
            object_text_features.append(self.get_clip_feature(clipembedder, object_phrase))
            action_text_features.append(self.get_clip_feature(clipembedder, action_phrase))

        for idx, (subject_box, object_box,
                subject_text_feature,
                object_text_feature,
                action_text_feature,) \
                in enumerate(zip(subject_boxes, object_boxes,
                                subject_text_features,
                                object_text_features,
                                action_text_features)):
            if idx >= max_objs:  # no more than max_obj
                break
            subject_boxes_tensor[idx] = torch.tensor(subject_box)
            object_boxes_tensor[idx] = torch.tensor(object_box)
            masks[idx] = 1
            if subject_text_feature is not None:
                subject_text_embeddings[idx] = subject_text_feature
                object_text_embeddings[idx] = object_text_feature
                action_text_embeddings[idx] = action_text_feature
                masks[idx] = 1

        batch_size = p.batch_size
        self.plug_ID.update_stages(strength, stage_one, stage_two)
        # do not repeat, will repeat at unet_signal
        self.plug_ID.update_objs(subject_boxes_tensor.unsqueeze(0), 
                                         object_boxes_tensor.unsqueeze(0), 
                                         masks.unsqueeze(0), 
                                         subject_text_embeddings.unsqueeze(0), 
                                         object_text_embeddings.unsqueeze(0),
                                         action_text_embeddings.unsqueeze(0),
                                         p.batch_size)
        self.plug_ID.attach_all()

        Infotext.write_infotext([interactdiffusion.InteractDiffusionData(
            enabled=enabled,
            scheduled_sampling_rate=stage_one,
            subject_boxes=subject_boxes,
            object_boxes=object_boxes,
            subject_phrases=subject_phrases,
            object_phrases=object_phrases,
            action_phrases=action_phrases
        )], p)

        return  #updates the UNet model with the pluggable modules, params and tensors. UNet can be used for downstream tasks in our case generating.


    def postprocess(self, *args):
        shared.pluggable_ID.detach_all()
        return
    
    # def DB_get_data(self):

    #     t
    

    # def train_DBooth(self):

    """
    TO DO:

    Integrate with dreambooth training
        1. Load model in the right format ( openai unet model)
        2. Write wrapper methods for training dreambooth
        3. Write wrapper methods for loading and feeding dataset
        4. Inference
    """


    
