import os
import re
import sys
import math
import random
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageOps, ImageDraw
from psd_tools import PSDImage


__all__ = ['PostGenerationEngine']
   

LAYER_BG   = 0
LAYER_IMG  = 1
LAYER_TEXT = 2

FONT_OFFSET = 20
FONT_SIZE = 300 + FONT_OFFSET
VERTICAL_SPACING = 60


def rgb2bgr(color):
    return list(reversed(color))


def tovec(layer, psd_size):
    w, h = psd_size
    l, t, r, b = layer.bbox
    l1 = 0 if l >= 0 else -l
    t1 = 0 if t >= 0 else -t
    img = layer.topil().crop((l1, t1, l1 + min(w, r - l), t1 + min(h, b - t)))
    pos = (l if l >= 0 else 0, t if t >= 0 else 0)
    return img, pos


def str_similarity(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0 for i, j in zip(str1, str2))


def match_font(fonts, font_name):
    scores = [str_similarity(font_name.value, str(font.getname()[0])) for font in fonts]
    return fonts[np.argmax(scores)]


def get_bbox_size(bbox):
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    

def get_primary_color(vec):
    vec = np.array(vec)
    return tuple(vec[vec[:,:,3] != 0].max(axis=0))


def textvec(text, font, fill, ratio, vertical=False, expanded=False):
    d = VERTICAL_SPACING if vertical else 0
    w = math.ceil(font.getlength(text) + d * (len(text) - 1))
    h = FONT_SIZE
    if expanded:
        if h * ratio > w:
            d = (h * ratio - font.getlength(text)) / (len(text) - 1) if len(text) > 1 else 0
            w = round(h * ratio)
    buf = Image.new("RGBA", (w, h) if not vertical else (h, w), (0, 0, 0, 0))
    draw = ImageDraw.Draw(buf)
    isascii = lambda s: len(s) == len(s.encode())
    for i, c in enumerate(text):
        if vertical and isascii(c):
            continue
        offset = font.getlength(text[:i+1]) - font.getlength(c) + i * d
        draw.text((offset, 0) if not vertical else (0, offset), c, font=font, fill=fill)
    if vertical:
        buf = buf.transpose(Image.Transpose.ROTATE_90)
        draw = ImageDraw.Draw(buf)
        for i, c in enumerate(text):
            if isascii(c):
                offset = font.getlength(text[:i+1]) - font.getlength(c) + i * d
                draw.text((offset, 0), c, font=font, fill=fill)
    return buf


def render_text(text, font, fill, size,
                centered=False, expanded=False, vertical=False,
                lines=1, lineWidths=None):
    if vertical:
        size = (size[1], size[0])
    
    segments = [item.strip() for item in re.split(r'[\r\n]', text) if item.strip()]

    if len(segments) == 1 and lines > 1:
        if lineWidths is None:
            lineWidths = [1]
        assert len(lineWidths) == lines
        s = segments[0]
        totalLength = len(s)
        weightSum = sum(lineWidths)
        cur = 0
        tmp = []
        for i in range(lines):
            now = round(totalLength * lineWidths[i] / weightSum) if i + 1 < lines else totalLength - cur
            if now > 0:
                tmp.append(s[cur:cur+now])
            cur += now
        segments = tmp
    
    lines = len(segments)
    h_max = size[1] / lines
    ratio = max((font.getlength(item) + (VERTICAL_SPACING * (len(item) - 1) if vertical else 0)) / FONT_SIZE for item in segments)
    w_max = h_max * ratio
    if w_max > size[0]:
        h_max *= size[0] / w_max
        w_max = size[0]
        ratio = w_max / h_max 
    if lines == 1 and size[0] > w_max:
        w_max = size[0]
        ratio = w_max / h_max
    parts = [textvec(item, font, fill, ratio=ratio, expanded=expanded or (lines == 1), vertical=vertical) for item in segments]
    margin = (size[1] - h_max * lines) / (lines - 1) if lines > 1 else 0
    res = Image.new("RGBA", size, (0, 0, 0, 0))
    for i, part in enumerate(parts):
        cur_x = (h_max + margin) * i
        cur_y = 0
        part = part.resize((round(part.width / part.height * h_max), round(h_max)))
        if centered:
            cur_y = (w_max - part.width) / 2
        res.paste(part, (round(cur_y), round(cur_x)))
    
    if vertical:
        res = res.transpose(Image.Transpose.ROTATE_270)

    return res


class ImageTemplate:
    name: str
    shape: Tuple[int, int]
    layers: List[dict]


def load_template(name, path):
    psd_name = list(filter(lambda name: name.endswith('.psd'), os.listdir(path)))[0]
    fonts_map = {
        name[:-4]: ImageFont.truetype(os.path.join(path, name), FONT_SIZE - FONT_OFFSET)
        for name in os.listdir(path)
        if name.lower().endswith('.ttf')
    }
    fonts = list(fonts_map.values())
    psd = PSDImage.open(os.path.join(path, psd_name))
    res = ImageTemplate()
    res.shape = psd.size
    res.layers = []
    for layer in psd:
        if layer.name.startswith('background layer'):
            vec, pos = tovec(layer, psd.size)
            res.layers.append({
                'kind': LAYER_BG,
                'key': layer.name.replace('background layer', 'bg'),
                'vec': vec,
                'pos': pos,
            })
        elif layer.name == 'embellishment layer':
            vec, pos = tovec(layer, psd.size)
            res.layers.append({
                'kind': LAYER_BG,
                'key': layer.name.replace('embellishment layer', 'ex'),
                'vec': vec,
                'pos': pos,
            })
        elif layer.name == 'image layer':
            if layer.is_group():
                for sub in layer:
                    if sub.has_vector_mask():
                        vec, pos = tovec(sub[0], psd.size)
                        mask_vec, mask_pos = tovec(sub.mask, psd.size)
                        res.layers.append({
                            'kind': LAYER_IMG,
                            'key': 'img',
                            'vec': vec,
                            'pos': pos,
                            'mask': {
                                'vec': mask_vec,
                                'pos': mask_pos,
                            }
                        })
                        break
            else:
                vec, pos = tovec(layer, psd.size)
                res.layers.append({
                    'kind': LAYER_IMG,
                    'key': 'img',
                    'vec': vec,
                    'pos': pos,
                    'mask': None
                })
        elif layer.name == 'title layer' or layer.name == 'subtitle layer':
            assert layer.kind == 'type'
            engine_data = layer.engine_dict
            segments = [item.strip() for item in re.split(r'[\r\n]', layer.text)]
            fontset = layer.resource_dict['FontSet']
            stylesheet = engine_data['StyleRun']['RunArray'][0]['StyleSheet']['StyleSheetData']
            font_name = fontset[stylesheet['Font']]['Name']
            res.layers.append({
                'kind': LAYER_TEXT,
                'key': layer.name.split(' ')[0],
                'font': match_font(fonts, font_name),
                'available_fonts': fonts_map,
                'fill': get_primary_color(tovec(layer, psd.size)[0]),
                'bbox': layer.bbox,
                'lines': len(segments),
                'expanded': len(list(filter(lambda s: s.find(' ') >= 0, segments))) > 0,
                'centered': engine_data['ParagraphRun']['RunArray'][0]['ParagraphSheet']['Properties'].get('Justification', 0) == 2,
                'vertical': engine_data['Rendered']['Shapes']['Children'][0]['Procession'] == 1,
                'lineWidths': [len(re.sub(r'\s*', '', item)) for item in segments],
            })
        elif layer.name == 'text layer':
            # ignore it
            pass
        else:
            raise Exception('Unsupported layer name %s' % layer.name)
    res.name = name
    return res


def load_templates(template_path):
    print('Loading templates ...', file=sys.stderr)
    res = {
        name: load_template(name, os.path.join(template_path, name))
        for name in os.listdir(template_path)
        if not name.startswith('.')
    }
    print('Successfully loaded %s templates:' % len(res), file=sys.stderr)
    for key in res.keys():
        print('- ' + key, file=sys.stderr)
    return res


def render(template: ImageTemplate,
           title: str,
           subtitle: str,
           image: Image=None,
           colors: Dict[str, str]={},
           fonts: Dict[str, str]={}) -> Image:
    print('Rendering new poster (title=%s) via template <%s>' % (title, template.name))
    res = Image.new("RGBA", template.shape, (0, 0, 0, 0))
    for layer in template.layers:
        kind = layer['kind']
        key = layer['key']
        if kind == LAYER_BG:
            pos = layer['pos']
            vec = layer['vec']
            if colors.get(key) is not None:
                color = colors.get(key)
                mask = Image.new("RGBA", vec.size, (0, 0, 0, 0))
                vec = np.array(vec)
                vec[:,:,:3] = 255
                vec = Image.fromarray(vec)
                mask.alpha_composite(vec)
                mask = mask.convert("L")
                buf = Image.new("RGBA", vec.size, color)
                empty = Image.new("RGBA", vec.size, (0, 0, 0, 0))
                vec = Image.composite(buf, empty, mask)
            res.alpha_composite(vec, pos)
        elif kind == LAYER_IMG:
            pos = layer['pos']
            vec = layer['vec']
            if image is not None:
                ratio = vec.size[0] / vec.size[1]
                ratio1 = image.size[0] / image.size[1]
                w = vec.size[0]
                h = w / ratio1
                if h < image.size[1]:
                    h = vec.size[1]
                    w = h * ratio1
                image = image.resize((round(w), round(h)))
                w = image.size[0]
                h = w / ratio
                if h > image.size[1]:
                    h = image.size[1]
                    w = h * ratio
                x = (image.size[0] - w) / 2
                y = (image.size[1] - h) / 2
                vec1 = image.crop((x, y, x+w, y+h)).resize(vec.size)
                vec = vec1
            mask = layer['mask']
            buf = Image.new("RGBA", template.shape, (0, 0, 0, 0))
            buf.paste(vec, pos)
            if mask is not None:
                empty = Image.new("RGBA", template.shape, (0, 0, 0, 0))
                buf2 = Image.new("L", template.shape, "black")
                buf2.paste(mask['vec'].convert("L"), mask['pos'])
                buf = Image.composite(empty, buf, ImageOps.invert(buf2))
            res.alpha_composite(buf)
        elif kind == LAYER_TEXT:
            text = title if key == 'title' else subtitle
            font = layer['font']
            available_fonts = layer['available_fonts']
            font_name = fonts.get(key)
            if font_name is not None and available_fonts.get(font_name) is not None:
                font = available_fonts[font_name]
            bbox = layer['bbox']
            fill = layer['fill']
            if colors.get(key) is not None:
                fill = colors.get(key)
            size = get_bbox_size(bbox)
            res.alpha_composite(render_text(
                text,
                font,
                fill,
                size,
                centered=layer['centered'],
                expanded=layer['expanded'],
                vertical=layer['vertical'],
                lines=layer['lines'],
                lineWidths=layer['lineWidths'],
            ), bbox[:2])

    return res


class PosterGenerationEngine:

    templates: Dict[str, ImageTemplate]
    corpus: Dict[str, List[str]]
    
    def __init__(self, template_path=None, corpus_path=None) -> None:
        if template_path is None:
            template_path = os.path.join(os.getcwd(), 'templates')
        
        if corpus_path is None:
            corpus_path = os.path.join(os.getcwd(), 'static/语料表.xlsx')

        self.templates = load_templates(template_path)
        df = pd.read_excel(corpus_path)
        self.corpus = {}
        for _, row in df.iterrows():
            label = row[0]
            candidates = [item.strip() for item in row[1].split('\n') if item.strip()]
            self.corpus[label] = candidates

    
    def generate_title(self, label: str) -> str:
        if self.corpus.get(label) is not None:
            return random.choice(self.corpus[label])
        return label


    def match_template(self, keyword: str) -> ImageTemplate:
        for item in self.templates.keys():
            keys = set(item.split(','))
            if keyword in keys:
                return self.templates[item]
        return None

    
    def render(self, *args, **kwargs) -> Image:
        return render(*args, **kwargs)
