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
import gradio as gr

from gradio import processing_utils


from modules.processing import StableDiffusionProcessing
gradio_compat = True
MAX_COLORS = 12
switch_values_symbol = '\U000021C5' # ⇅

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"] and type(x) != dict:
            decode_image = processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


def draw_box(boxes=[], texts=[], img=None, width=512, height=512):
    if len(boxes) == 0 and img is None:
        return None

    if img is None:
        img = Image.new('RGB', (height, width), (255, 255, 255))
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=18)

    for bid, box in enumerate(boxes):
        hid = bid // 2
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[hid % len(colors)], width=4)
        anno_text = texts[hid * 3] if bid % 2 == 0 else texts[hid * 3 + 2]
        draw.rectangle(
            [box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]],
            outline=colors[hid % len(colors)], fill=colors[hid % len(colors)], width=4)
        draw.text([box[0] + int(font.size * 0.2), box[3] - int(font.size * 1.2)], anno_text, font=font,
                  fill=(255, 255, 255))
        if bid % 2 == 1:
            act_text = texts[hid * 3 + 1]
            draw.line([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                       ((boxes[bid - 1][0] + boxes[bid - 1][2]) / 2, (boxes[bid - 1][1] + boxes[bid - 1][3]) / 2)],
                      fill=colors[hid % len(colors)], width=3)
            _, _, w, h = draw.textbbox((0, 0), act_text, font=font)
            text_x = (min(box[0], boxes[bid - 1][0])+max(box[2], boxes[bid - 1][2]) - w)/2
            text_y = (min(box[1], boxes[bid - 1][1])+max(box[3], boxes[bid - 1][3]) - h)/2
            # logging.warn(f"y0: {text_y + int(font.size * 1.2)}, y1: {text_y}")
            draw.rectangle(
                (text_x, text_y,
                 text_x + int((len(act_text) + 0.8) * font.size * 0.6), text_y + int(font.size * 1.2)),
                outline=colors[hid % len(colors)], fill="white", width=1)
            draw.text((text_x, text_y),
                      act_text, align="center",
                      font=font, fill=colors[hid % len(colors)])
    return img


def draw(task, input, grounding_texts, new_image_trigger, state):
    if type(input) == dict:
        image = input['image']
        width, height = image.shape[:2]
        mask = input['mask']
    else:
        mask = input

    image_scale = 1.0

    if mask is None:
        return [None, new_image_trigger, image_scale, state]

    if mask.ndim == 3:
        mask = mask[..., 0]


    mask = binarize(mask)

    if type(mask) != np.ndarray:
        mask = np.array(mask)

    if mask.sum() == 0 and task != "Grounded Inpainting":
        if "example" in state and state['example']:
            # print(state)
            pass
        else:
            state = {}

    if task != 'Grounded Inpainting':
        image = None
    else:
        image = Image.fromarray(image)

    if 'boxes' not in state:
        state['boxes'] = []

    if 'masks' not in state or len(state['masks']) == 0:
        state['masks'] = []
        last_mask = np.zeros_like(mask)
    else:
        last_mask = state['masks'][-1]

    if type(mask) == np.ndarray and mask.size > 1:
        diff_mask = mask - last_mask
    else:
        diff_mask = np.zeros([])

    if diff_mask.sum() > 0:
        x1x2 = np.where(diff_mask.max(0) != 0)[0]
        y1y2 = np.where(diff_mask.max(1) != 0)[0]
        y1, y2 = y1y2.min(), y1y2.max()
        x1, x2 = x1x2.min(), x1x2.max()

        if (x2 - x1 > 5) and (y2 - y1 > 5):
            state['masks'].append(mask.copy())
            state['boxes'].append((x1, y1, x2, y2))

    grounding_texts = [y for x in grounding_texts.split(';') for y in x.strip().split("=") if y != ""]
    grounding_texts = [x for x in grounding_texts if len(x) > 0]
    if 2 * len(grounding_texts) < 3 * len(state['boxes']):
        grounding_texts += [(f'Obj. {bid//3 * 2 + 1}' if bid % 3 == 0 else f'Obj. {bid//3 * 2 + 2}')
                            if bid % 3 != 1 else f'Act. {bid//3 + 1}'
                            for bid in range(len(grounding_texts), round(len(state['boxes'])*3/2))]
        
    box_image = draw_box(state['boxes'], grounding_texts, image, width, height)

    if box_image is not None and state.get('inpaint_hw', None):
        inpaint_hw = state['inpaint_hw']
        box_image_resize = np.array(box_image.resize((inpaint_hw, inpaint_hw)))
        original_image = state['original_image'].copy()
        box_image = sized_center_fill(original_image, box_image_resize, inpaint_hw, inpaint_hw)

    return [box_image, new_image_trigger, image_scale, state]

def clear(task, sketch_interaction_pad_trigger, state, switch_task=False):
    if task != 'Grounded Inpainting':
        sketch_interaction_pad_trigger = sketch_interaction_pad_trigger + 1

    state = {}
    return [None, sketch_interaction_pad_trigger, None, 1.0] + [state]


shared.pluggable_ID = None


class Script(scripts.Script):
    def __init__(self):
        model_path = pathlib.Path(__file__).parent.parent / 'models' / 'ext_interactdiff_v1.2.pth'
        interactdiffusion_state_dict = torch.load(str(model_path), map_location=shared.device)
        if not shared.pluggable_ID:
            shared.pluggable_ID = PluggableInteractDiffusion(shared.sd_model.model.diffusion_model,interactdiffusion_state_dict)
        return

    def title(self):
        return "InteractDiffusion extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        infotext = Infotext()
        process_script_params = []
        with gr.Group() as group_interactdiffusion_root:
            with gr.Accordion("InteractDiffusion", open=False):
                enabled = gr.Checkbox(value=False, label="Enabled")
                with gr.Row():
                    with gr.Column(scale=4):
                        sketch_interaction_pad_trigger = gr.Number(value=0, visible=False)
                        sketch_interaction_pad_resize_trigger = gr.Number(value=0, visible=False)
                        init_white_trigger = gr.Number(value=0, visible=False)
                        image_scale = gr.Number(value=0, elem_id="image_scale", visible=False)
                        new_image_trigger = gr.Number(value=0, visible=False)

                        task = gr.Radio(
                            choices=["Grounded Generation", 'Grounded Inpainting'],
                            type="value",
                            value="Grounded Generation",
                            label="Task",
                            visible=False
                        )
                        grounding_instruction = gr.Textbox(
                            label="Grounding instruction ('subject=action=object' separated by semicolon)",
                        )
                        strength = gr.Slider(label="Strength", labminimum=0.0, maximum=2.0, value=1.0, step=0.01, interactive=True, visible=False)
                        stage_one = gr.Slider(label="Schedule Sampling (Amount of Control), reduce to 0.3 while using LoRA", labminimum=0.0, maximum=1.0, value=0.8, step=0.05, interactive=True)
                        stage_two = gr.Slider(label="Schedule sampling decay", minimum=0.0, maximum=1.0, value=0.0, step=0.01, interactive=True, visible=False)

                        with gr.Row():
                            clear_btn = gr.Button(value='Create/Clear Drawing Canvas', elem_id="clear_btn")
                        
                        with gr.Row():
                            sketch_interaction_pad = ImageMask(label="Sketch Pad", elem_id="img2img_image")
                            out_imagebox = gr.Image(type="pil", label="Parsed Sketch Pad")
                        
                        #################################################
                        # Create new canvas with custom size
                        # Current status: buggy, hence comment out first
                        #################################################
                        # def create_canvas(h, w):
                        #     return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255
                        # with gr.Row():
                        #     # with gr.Column():
                        #     canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512,
                        #                                 step=64)
                        #     canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512,
                        #                                 step=64)

                        #     if gradio_compat:
                        #         canvas_swap_res = ToolButton(value=switch_values_symbol)
                        #         canvas_swap_res.click(lambda w, h: (h, w), inputs=[canvas_width, canvas_height],
                        #                               outputs=[canvas_width, canvas_height])
                        #     create_button = gr.Button(value="Create blank canvas with other size")
                        # create_button.click(fn=create_canvas, inputs=[canvas_height, canvas_width],
                        #                     outputs=[sketch_interaction_pad])

        state = gr.State({})

        class Controller:
            def __init__(self):
                self.calls = 0
                self.tracks = 0
                self.resizes = 0
                self.scales = 0

            def init_white(self, init_white_trigger):
                self.calls += 1
                # [sketch_interaction_pad, image_scale, init_white_trigger]
                return np.ones((512, 512), dtype='uint8') * 255, 1.0, init_white_trigger + 1

            def resize_centercrop(self, state):
                self.resizes += 1
                image = state['original_image'].copy()
                inpaint_hw = int(0.9 * min(*image.shape[:2]))
                state['inpaint_hw'] = inpaint_hw
                image_cc = center_crop(image, inpaint_hw)
                # print(f'resize triggered {self.resizes}', image.shape, '->', image_cc.shape)
                return image_cc, state

            def resize_masked(self, state):
                self.resizes += 1
                image = state['original_image'].copy()
                inpaint_hw = int(0.9 * min(*image.shape[:2]))
                state['inpaint_hw'] = inpaint_hw
                image_mask = sized_center_mask(image, inpaint_hw, inpaint_hw)
                state['masked_image'] = image_mask.copy()
                # print(f'mask triggered {self.resizes}')
                return image_mask, state

            def switch_task_hide_cond(self, task):
                cond = False
                if task == "Grounded Generation":
                    cond = True
                return gr.Checkbox.update(visible=cond, value=False), gr.Image.update(value=None,
                                                                                      visible=False), gr.Slider.update(
                    visible=cond), gr.Checkbox.update(visible=(not cond), value=False)

        controller = Controller()
        sketch_interaction_pad.edit(
            draw,
            inputs=[task, sketch_interaction_pad, grounding_instruction, sketch_interaction_pad_resize_trigger, state],
            outputs=[out_imagebox, sketch_interaction_pad_resize_trigger, image_scale, state],
            queue=True,
        )
        grounding_instruction.change(
            draw,
            inputs=[task, sketch_interaction_pad, grounding_instruction, sketch_interaction_pad_resize_trigger, state],
            outputs=[out_imagebox, sketch_interaction_pad_resize_trigger, image_scale, state],
            queue=False,
        )
        clear_btn.click(
            clear,
            inputs=[task, sketch_interaction_pad_trigger, state],
            outputs=[sketch_interaction_pad, sketch_interaction_pad_trigger, out_imagebox, image_scale, state],
            queue=False)
        sketch_interaction_pad_trigger.change(
            controller.init_white,
            inputs=[init_white_trigger],
            outputs=[sketch_interaction_pad, image_scale, init_white_trigger],
            queue=False)
        sketch_interaction_pad_resize_trigger.change(
            controller.resize_masked,
            inputs=[state],
            outputs=[sketch_interaction_pad, state],
            queue=False)
        sketch_interaction_pad_resize_trigger.change(
            None,
            None,
            sketch_interaction_pad_resize_trigger,
            _js=rescale_js,
            queue=False)
        init_white_trigger.change(
            None,
            None,
            init_white_trigger,
            _js=rescale_js,
            queue=False)
        process_script_params.append(enabled)
        process_script_params.extend([
            task,  grounding_instruction, sketch_interaction_pad,
            state,
            strength, stage_one, stage_two
        ])
        #############################################################
        # currently, it is still not suitable to use infotext,      #
        # because there is no native support for bounding box input #
        #############################################################
        # infotext.register_unit(0, )

        # self.infotext_fields = infotext.infotext_fields
        # self.paste_field_names = infotext.paste_field_names

        return process_script_params

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

    def process(self, p: StableDiffusionProcessing, *args, **kwargs):
        enabled, task, grounding_texts, sketch_interaction_pad, state, \
            strength, stage_one, stage_two = args
        if not enabled:
            return
        if 'boxes' not in state:
            state['boxes'] = []
        sketch_image = sketch_interaction_pad['image']
        boxes = state['boxes']
        height_width_arr = np.array([sketch_image.shape[1], sketch_image.shape[0], sketch_image.shape[1], sketch_image.shape[0]])
        grounding_texts = [y for x in grounding_texts.split(';') for y in x.strip().split("=") if y != ""]
        subject_phrases = grounding_texts[0::3]
        action_phrases = grounding_texts[1::3]
        object_phrases = grounding_texts[2::3]
        subject_boxes = boxes[0::2]
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
        shared.pluggable_ID.update_stages(strength, stage_one, stage_two)
        # do not repeat, will repeat at unet_signal
        shared.pluggable_ID.update_objs(subject_boxes_tensor.unsqueeze(0), 
                                         object_boxes_tensor.unsqueeze(0), 
                                         masks.unsqueeze(0), 
                                         subject_text_embeddings.unsqueeze(0), 
                                         object_text_embeddings.unsqueeze(0),
                                         action_text_embeddings.unsqueeze(0),
                                         p.batch_size)
        shared.pluggable_ID.attach_all()

        Infotext.write_infotext([interactdiffusion.InteractDiffusionData(
            enabled=enabled,
            scheduled_sampling_rate=stage_one,
            subject_boxes=subject_boxes,
            object_boxes=object_boxes,
            subject_phrases=subject_phrases,
            object_phrases=object_phrases,
            action_phrases=action_phrases
        )], p)

        return

    def postprocess(self, *args):
        shared.pluggable_ID.detach_all()
        return

def infotext_pasted(infotext, params):
    logging.warn("from InteractDiffusion!")
    logging.warn(f"infotext: {infotext}")
    logging.warn(f"params: {params}")

# infotext is still buggy
# script_callbacks.on_infotext_pasted(Infotext.on_infotext_pasted)


