import cv2
import numpy as np

from .layout_util import lines_to_regions, boxes_to_lines
from .polygon_util import polygon_area, get_top_point
from .textsearch_util import noisy_words, calculate_sim_score


def process_text_of_interest(img, boxes, texts, unit='region', N=5, txt_tk=2, txt_scale=1):
    toi_text = []
    if unit == None or unit == 'word':
        word_areas = [polygon_area(box) for box in boxes]
        area_scores = [ round((area/img.size)*1000, 10) for area in word_areas]
        sidx = np.argsort(area_scores)[::-1][:len(area_scores)]
        boxes = np.array(boxes) if isinstance(boxes, list) else boxes
        return visualize_tois(texts, boxes, sidx, img, N)

    # Combining word-level -> line-level
    line_polys, line_texts, line_areas, _ = boxes_to_lines(boxes, texts)
    line_polys = [poly[:,0,:] for poly in line_polys]

    ## ToI by line area
    if unit == 'line':
        area_scores = [ round((area/img.size)*1000, 10) for area in line_areas]
        sidx = np.argsort(area_scores)[::-1][:len(area_scores)]
        return visualize_tois(line_texts, line_polys, sidx, img, N)

    # Combining line-level -> region-level
    #line_results = [poly[:,0,:] for poly in line_polys]
    region_polys, region_texts, region_areas, _ = lines_to_regions(line_polys, line_texts, line_areas)
    region_polys = [poly[:,0,:] for poly in region_polys]

    if unit == 'region':
        region_area_scores = [round((area/img.size)*1000, 10) for area in region_areas]
        sidx = np.argsort(region_area_scores)[::-1][:len(region_area_scores)]
        return visualize_tois(region_texts, region_polys, sidx, img, N)



def get_toi_all_units(img, polygons, decoded_txts):
    all_toi_texts = []
    for unit in ['word', 'line', 'region']:
        img, toi_texts = process_text_of_interest(img, polygons, decoded_txts, unit=unit)    
        #toi_texts = [' '.join(ls) for ls in toi_texts] if unit == 'region' else toi_texts
        all_toi_texts.extend(toi_texts)
    return all_toi_texts


def get_toi_tops(place_texts, toi_texts, K=3, sim_cutoff=0.5):
    results = {}
    noise = noisy_words(place_texts)
    for text in toi_texts:
        text_scores = {}
        text_results = {}
        for pid, pname in place_texts.items():
            text_scores[pid] = calculate_sim_score(text, pname, noise)
        sim = np.argsort(-np.array(list(text_scores.values())))

        K_counter = 0
        i = 0
        prev_score = list(text_scores.values())[sim[i]]
        while K_counter < K and i < len(sim) and sim[i] < len(text_scores):
            score = list(text_scores.values())[sim[i]]
            if score < sim_cutoff:
                prev_score = score
                i += 1
                continue
            if score == prev_score:    
                pid = list(text_scores.keys())[sim[i]]
                pname = place_texts[pid]
                text_results[pid] = (pname,score)
            else:
                K_counter += 1
            prev_score = score
            i += 1
        results[text] = text_results
    return results



def get_final_top(top_matches, K=3):
    all_top = {}
    for text in top_matches.keys():
        for k, v in top_matches[text].items():
            score = v[1]
            if k not in all_top.keys():
                all_top[k] = score
            elif all_top[k] < score:
                all_top[k] = score

    pid_list = list(all_top.keys())
    score_list = list(all_top.values())
    ranking = np.argsort(-np.array(score_list))

    final_top = []
    for i in range(K):
        if len(ranking) > i and len(score_list) > ranking[i]:
            pid = pid_list[ranking[i]]
            score = score_list[ranking[i]]
            final_top.append(pid)
    
    #if more places have the same score, iclude them to too
    i = K 
    while len(ranking) > i and len(score_list) > ranking[i] and score_list[ranking[i-1]] == score_list[ranking[i]]:
        pid = pid_list[ranking[i]]
        score = score_list[ranking[i]]
        final_top.append(pid)
        i+=1

    return final_top


def visualize_tois(texts, boxes, sidx, img, N):
    toi_text = []
    for i, (poly,txt) in enumerate(zip(boxes, texts)):
            if i in sidx[:min(N,len(sidx))]: #top x
                cv2.polylines(img,[poly], True, (255,255,255), 2) #20
                cv2.polylines(img,[poly], True, (0,0,0), 1) #2
                #cv2.putText(img, text=txt, org=(poly[0,0], poly[0,1]),fontFace=cv2.FONT_HERSHEY_TRIPLEX , fontScale=font_size, color=(255,0,0), thickness=4, lineType=cv2.LINE_AA)
                idx = get_top_point(poly)
                draw_text(img, text=txt, font=cv2.FONT_HERSHEY_TRIPLEX, pos=(poly[idx,0], poly[idx,1]),font_scale=1,font_thickness=2, text_color=(0, 0, 0),text_color_bg=(255, 255, 255))
                toi_text.append(txt)
    return img, toi_text

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def filter_by_area(img, texts, boxes, scores, recs):
    noisy_text = ["sale", "cafe", "harajuku"]
    fl_boxes, fl_texts, fl_scores, fl_recs = [], [], [],[]
    for i in range(len(boxes)):
        if texts[i].lower() in noisy_text:
            continue
        boxes[i] = boxes[i].tolist() if not isinstance(boxes[i], list) else boxes[i]
        area_score = (polygon_area(boxes[i])/img.size)*1000
        if area_score >= 0.1:
            fl_boxes.append(boxes[i])
            fl_texts.append(texts[i])
            fl_scores.append(scores[i])
            fl_recs.append(recs[i])
    assert len(fl_texts) == len(fl_boxes) == len(fl_scores) == len(fl_recs)
    return fl_texts, fl_boxes, fl_scores, fl_recs