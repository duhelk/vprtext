import cv2
from torch import Tensor
from .common_util import *
from .textspotter_util import TextSpotter

from .common_util import encode_recognition, rearrange_points, create_out_file, sample_polygon

from davarocr.davar_common.apis import inference_model, init_model

class MANGOSpotter(TextSpotter):
    def __init__(self) -> None:
        self.config_file = './DAVAR-Lab-OCR/demo/text_spotting/mango/configs/mango_r50_ete_finetune_tt.py'
        self.checkpoint_file = './DAVAR-Lab-OCR/demo/text_spotting/mango/log/checkpoint/res50_ete_finetune_tt.pth' 
        self.predictor = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        self.cfg = self.predictor.cfg

    def predict(self, img_path, out_file=None):
        predictions = inference_model(self.predictor, img_path)[0]
        texts = predictions['texts']
        polys = predictions['points']
        scores = list(predictions['cate_weights']) ## Verify
        recs =  [encode_recognition(txt) for txt in texts]

        if out_file != None:
            create_out_file(cv2.imread(img_path), texts, polys, out_file)

        polygons = []
        for p in polys: 
            pts = [[p[i], p[i+1]] for i in range(0, len(p), 2)]
            box = rearrange_points(pts)
            polygons.append(box)
            
        return texts, Tensor(polygons), scores, recs


class TextPerceptronSpotter(TextSpotter):
    def __init__(self) -> None:
        self.config_file = './DAVAR-Lab-OCR/demo/text_spotting/text_perceptron_spot/configs/tp_r50_e2e_finetune_tt.py'
        self.checkpoint_file = './DAVAR-Lab-OCR/demo/text_spotting/text_perceptron_spot/log/checkpoint/tp_r50_e2e_finetune.pth' 
        self.predictor = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        self.cfg = self.predictor.cfg

    def predict(self, img_path, out_file=None):
        predictions = inference_model(self.predictor, img_path)[0]
        #print("="*5, predictions)
        texts = predictions['texts']
        polys = predictions['points']
        scores = [0 for x in range(len(predictions['texts']))]
        recs =  [encode_recognition(txt) for txt in texts]

        if out_file != None:
            create_out_file(cv2.imread(img_path), texts, polys, out_file)

        polygons = []
        for p in polys: 
            pts = [[p[i], p[i+1]] for i in range(0, len(p), 2)]
            box = rearrange_points(pts)
            polygons.append(box)
            
        return texts, Tensor(polygons), scores, recs


class MaskRCNNSpotter(TextSpotter):
    def __init__(self) -> None:
        self.config_file = './DAVAR-Lab-OCR/demo/text_spotting/mask_rcnn_spot/configs/mask_rcnn_r50_conv6_e2e_finetune_tt.py'
        self.checkpoint_file = './DAVAR-Lab-OCR/demo/text_spotting/mask_rcnn_spot/log/checkpoint/mask_rcnn_r50_conv6_e2e_finetune.pth' 
        self.predictor = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        self.cfg = self.predictor.cfg

    def predict(self, img_path, out_file=None):
        predictions = inference_model(self.predictor, img_path)[0]
        texts = predictions['texts']
        polys = predictions['points']
        scores = [0 for x in range(len(predictions['texts']))]
        #list(predictions['cate_weights']) ## Verify
        recs =  [encode_recognition(txt) for txt in texts]

        if out_file != None:
            create_out_file(cv2.imread(img_path), texts, polys, out_file)

        polygons = []
        for p in polys: 
            pts = [[p[i], p[i+1]] for i in range(0, len(p), 2)]
            box = sample_polygon(pts) #format_polygon(pts)
            polygons.append(box)
        
        return texts, polygons, scores, recs


class DLDSpotter(TextSpotter):
    def __init__(self) -> None:
        self.config_file = './DAVAR-Lab-OCR/demo/text_spotting/dld/configs/mask_rcnn_distill.py'
        self.checkpoint_file = './DAVAR-Lab-OCR/demo/text_spotting/dld/mask_rcnn_res50_distill_y_0.1-4a9000b6.pth' 
        self.predictor = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        self.cfg = self.predictor.cfg

    def predict(self, img_path, out_file=None):
        predictions = inference_model(self.predictor, img_path)[0]
        texts = predictions['texts']
        polys = predictions['points']
        scores = [0 for x in range(len(predictions['texts']))]
        recs =  [encode_recognition(txt) for txt in texts]

        if out_file != None:
            create_out_file(cv2.imread(img_path), texts, polys, out_file)

        polygons = []
        for p in polys: 
            pts = [[p[i], p[i+1]] for i in range(0, len(p), 2)]
            box = rearrange_points(pts)
            polygons.append(box)
            
        return texts, Tensor(polygons), scores, recs



