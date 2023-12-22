import pickle
from torch import Tensor
import os
import cv2
import numpy as np
from shapely.geometry import Polygon

def get_char_list(lang='en'):
    if lang == 'en':
        CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
    elif lang == 'ch':
        CTLABELS = pickle.load(open('../chn_cls_list.txt', 'rb'))
    elif lang == 'jp':
        CTLABELS = pickle.load(open('../jp_chr_list.txt', 'rb'))
    voc_size = len(CTLABELS) + 1
    return CTLABELS, voc_size


CTLABELS, voc_size = get_char_list('en')

def decode_recognition(rec):
    s = ''
    rec = Tensor.tolist(rec)
    for c in rec:
        c = int(c)
        if c <  voc_size - 1:
            if voc_size == 96:
                s += CTLABELS[c]
            else:
               s += chr(CTLABELS[c])
        elif c == voc_size -1:
            s += u'口'
    return s

def encode_recognition(txt):
    rec = [96] * 25
    for i,c in enumerate(txt):
        if i >= 25:
            break
        try:
            rec[i] = CTLABELS.index(c.strip())
        except:
            rec[i] = 96
    return rec


def create_out_file(img, texts, polys, outfile):
    out_dir = os.path.dirname(os.path.realpath(outfile))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    img_draw = img.copy()
    for j, box in enumerate(polys):
        for i in range(0, len(box), 2):
            cv2.line(img_draw, (int(box[i]), int(box[i+1])),(int(box[(i+2)%len(box)]),  int(box[(i+3)%len(box)])), (255,0,0),1)
        img_draw = add_text(img_draw, str(texts[j]), int(box[i]-5), int(box[i+1]-5), (0, 0, 0), textSize=15)
    cv2.imwrite(outfile, img_draw)


def add_text(img, text, left, top, textColor, textSize):
    cv2.putText(img, text=text, org=(max(left,0), max(top,0)),fontFace=cv2.FONT_HERSHEY_TRIPLEX , fontScale=1, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def rearrange_points(pts):
    min_xy = 0
    for i in range(len(pts)):
        if sum(pts[i]) < sum(pts[min_xy]):
            min_xy = i

    pts = pts[min_xy:] + pts[0:min_xy]
    return pts

def sample_polygon(polygon):
    if len(polygon)%4 == 0:
        return polygon
    n_points = ((len(polygon)//4)+1)*4
    polygon = cv2.convexHull(np.concatenate(([polygon])))
    polygon = [co[0] for co in polygon.tolist()]
    polygon = Polygon(polygon)
    # Sampling the coordinates
    sample_coords = [polygon
                     .exterior
                     .interpolate(i/n_points, 
                                  normalized=True) for i in range(n_points)]

    coords = [[int(co.x), int(co.y)]for co in sample_coords]
    coords = rearrange_points(coords)
    return coords 
