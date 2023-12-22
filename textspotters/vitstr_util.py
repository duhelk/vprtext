from pydoc import text
from statistics import mode
from sys import modules

#from .utils import TokenLabelConverter
import argparse
from unittest import result
from unittest.mock import DEFAULT
from cv2 import imwrite
import torch
import os
from collections import OrderedDict
import string
import cv2
import pandas as pd
import numpy as np
from numpy import array
from numpy import int32
from .toi_util import boxes_to_lines
from .common_util import *

#import sys
#sys.path.append("./deep-text-recognition-benchmark")
from modules.vitstr import create_vitstr
from modules.transformation import TPS_SpatialTransformerNetwork
from infer_utils import NormalizePAD, ViTSTRFeatureExtractor, TokenLabelConverter
from model import Model

models = {'tiny':'vitstr_tiny_patch16_224',
          'small': 'vitstr_small_patch16_224',
          'base':'vitstr_base_patch16_224'}

saved_models = {
    'base_aug':'./deep-text-recognition-benchmark/vitstr_base_patch16_224_aug.pth',
    'small':'./deep-text-recognition-benchmark/vitstr_small_patch16_224.pth'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description='ViTSTR evaluation')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=224, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=224, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--input-channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--model', default="vitstr_small_patch16_224_aug_infer.pth", help='ViTSTR model')
    parser.add_argument('--gpu', action='store_true', help='use gpu for model inference')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    # parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    # parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    # parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    # parser.add_argument('--Transformer', type=str, required=True, help='Prediction stage. CTC|Attn')

    args = parser.parse_args(args=[])
    return args


class ViTRecognizer():

    def __init__(self) -> None:
        opt = get_args()
        opt.character = string.printable[:-6]
        self.converter = TokenLabelConverter(opt)
        opt.num_class = len(self.converter.character)
        if opt.rgb:
            opt.input_channel = 3

        if opt.Transformation == 'TPS':
            self.transformation = TPS_SpatialTransformerNetwork(F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)

        self.transformer_model = models['base']
        self.checkpoint_path = saved_models['base_aug']
        self.model= create_vitstr(num_tokens=opt.num_class, model=self.transformer_model)
        state_dict = self.load_state_dict(self.checkpoint_path, use_ema=False)
        self.model.load_state_dict(state_dict, strict=False)
        if torch.cuda.is_available():
            self.transformation.cuda()
            self.model.cuda()
        self.extractor = ViTSTRFeatureExtractor()

    def load_state_dict(self, checkpoint_path, use_ema=False):
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[14:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        return state_dict

    def crop_box(self,image_path, box, out_file):
        all_x = [int(coord[0]) for coord in box]
        all_y = [int(coord[1]) for coord in box]
        min_x = min(all_x) - 10
        max_x = max(all_x) + 10
        min_y = min(all_y) - 10
        max_y = max(all_y) + 10
        img = cv2.imread(image_path)
        crop_img = img[min_y:max_y, min_x:max_x]
        if crop_img.size == 0:
            return 0
        cv2.imwrite(out_file, crop_img)
    
    def recognize(self, image_path, out_file=None):
        img = self.extractor(image_path).to(device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(img, seqlen=self.converter.batch_max_length)
            _, pred_index = predictions.topk(1, dim=-1, largest=True, sorted=True)
            pred_index = pred_index.view(-1, self.converter.batch_max_length)
            length_for_pred = torch.IntTensor([self.converter.batch_max_length - 1] )
            pred_str = self.converter.decode(pred_index[:, 1:], length_for_pred)
            pred_EOS = pred_str[0].find('[s]')
            text = pred_str[0][:pred_EOS]
            rec = pred_index[:, 1:].tolist()[0]
            for i, x in enumerate(rec):
                if x == 1:
                    rec[i] = 96
            return rec, text

    def detect(self,image_path, unit=None):
        query_id = image_path.split('/')[-1].split('.')[0]   
        if 'NTXT' in query_id:
            res_csv = "/media/HD2/Workspace/Phase2/TextSpotting/Results/outABCNet_line/RES_NTXT.csv"
        elif 'TTXT' in query_id:
            res_csv = "/media/HD2/Workspace/Phase2/TextSpotting/Results/outABCNet_line/RES_TTXT.csv"
        result_df = pd.read_csv(res_csv, index_col='query')

        polys = result_df.loc[query_id, 'polys']
        if isinstance(polys, str):
            polys = eval(result_df.loc[query_id, 'polys'])
        else:
            polys = eval(result_df.loc[query_id, 'polys'].values[0])
        if unit == 'line':
            texts = ['' for x in range(len(polys))]
            line_polys, _, _, _ = boxes_to_lines(polys, texts)
            boxes = []
            for b in line_polys: 
                pts = [co[0] for co in b.tolist()]
                box = sample_polygon(pts) 
                boxes.append(box)
        else:
            boxes = polys
        return boxes


    def predict(self, image_path, out_file=None, unit=None):
        boxes = self.detect(image_path, unit)
        query_id = image_path.split('/')[-1].split('.')[0]  
        
        texts = []
        recs = []
        polygons = []
        for i, box in enumerate(boxes):
            outfile = os.path.join('temp', f'{query_id}_{str(i)}.jpeg')
            out = self.crop_box(image_path, box, outfile)
            if out != 0:
                rec, txt = self.recognize(outfile)
                recs.append(rec)
                texts.append(txt)
                polygons.append(box)

        scores = [0 for x in range(len(boxes))]

        
        return texts, polygons, scores, recs 

    
