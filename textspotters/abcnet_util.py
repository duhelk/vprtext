
import numpy as np
import os
import cv2
#import sys
#sys.path.append('./AdelaiDet')

from AdelaiDet.predictor.predictor import Predictor, setup_config

from .common_util import decode_recognition
from .textspotter_util import TextSpotter

class ABCNetSpotter(TextSpotter):
    def __init__(self) -> None:
        self.config_file = "./AdelaiDet/configs/BAText/TotalText/v2_attn_R_50.yaml"
        self.weights_file = "./AdelaiDet/weights/model_v2_totaltext.pth"
        self.predictor = Predictor(setup_config(self.config_file, self.weights_file))

    def predict(self, img_path, out_file=None):
        image = cv2.imread(img_path)
        predictions, visualized_output = self.predictor.run_on_image(image)
        if out_file != None:
            out_dir = os.path.dirname(os.path.realpath(out_file))
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            visualized_output.save(out_file)
        texts = [decode_recognition(rec) for rec in predictions["instances"].recs]
        recs = predictions["instances"].recs.tolist()
        polygons = [self.bezier_to_polygon(b.tolist()) for b in predictions["instances"].beziers]
        scores = predictions["instances"].scores.tolist()
        return texts, polygons, scores, recs

    def bezier_to_polygon(self, b):
        poly = []
        for i in range(0, len(b), 2):
            poly.append([b[i], b[i+1]])
        return np.array(poly, np.int32)
