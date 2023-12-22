from .common_util import *

from mmocr.utils.ocr import MMOCR
from .textspotter_util import TextSpotter

class MMOCRSpotter(TextSpotter):

    def __init__(self, det_model='DRRG', rec_model='SEG') -> None:
        self.predictor = MMOCR(det=det_model, recog=rec_model, kie='', device='cpu', config_dir='./mmocr/configs/')

    def predict(self, image, out_file=None):
        predictions = self.predictor.readtext(image,details=True, output=out_file)[0]['result']
        texts, polygons, scores, recs = [], [], [], []
        for pred in predictions:
            texts.append(pred['text'])
            box = pred['box']
            polygons.append([[box[i], box[i+1]] for i in range(0, len(box), 2)])
            scores.append(pred['text_score'])
            recs.append(encode_recognition(pred['text']))
        return texts, polygons, scores, recs
