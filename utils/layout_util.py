
import cv2
import numpy as np

from .polygon_util import polygon_area

def boxes_to_lines(boxes, texts): 
    box_list = []
    for i, (bez, txt) in enumerate(zip(boxes,texts)):
        all_x = [int(coord[0]) for coord in bez]
        all_y = [int(coord[1]) for coord in bez]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        tl, bl, tr, br = 0, int(len(bez)-1), int(len(bez)/2-1), int(len(bez)/2)
        l_hi = abs(bez[tl][1] - bez[bl][1])
        r_hi = abs(bez[tr][1] - bez[br][1])
        height = max( l_hi , r_hi) #0.5 * (l_hi + r_hi)
        width = max(abs(bez[tl][0] - bez[tr][0]), abs(bez[bl][0] - bez[br][0]))
        l_cen = (0.5 *(bez[tl][0] + bez[bl][0]), 0.5 *(bez[tl][1] + bez[bl][1]))
        r_cen = (0.5 *(bez[tr][0] + bez[br][0]), 0.5 *(bez[tr][1] + bez[br][1]))
        box_list.append([txt.lower(),bez, min_x, max_x, min_y, max_y, l_hi, r_hi, height, width, l_cen, r_cen, 0]) # last element indicates group

    _tx, _bx, _minx, _maxx, _miny, _maxy, _lhi, _rhi, _height, _width, _lcen, _rcen, _line  = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12

    line_id = 1
    line_str ={}
    line_box = {}
    while len([box for box in box_list if box[_line]==0]) > 0:
        box_line0 = [box for box in box_list if box[_line]==0] # group0 = non-group
        # new group
        if len([box for box in box_list if box[_line]==line_id]) == 0:
            box_line0[0][_line] = line_id # assign first box to form new group
            line_str[line_id] = box_line0[0][_tx] + ' '
            line_box[line_id] = box_line0[0][_bx]
        # try to add group
        else:
            current_line_boxes = [box for box in box_list if box[_line]==line_id]
            add_box = False
            for bx0 in box_line0:
                for lbx in current_line_boxes:
                    if bx0[_minx] > lbx[_minx]:
                        bx_l, bx_r = lbx, bx0
                        lbx_left = True
                        thres_diff = 0.5 * lbx[_rhi]
                    else:
                        bx_l, bx_r = bx0, lbx 
                        lbx_left = False
                        thres_diff = 0.5 * lbx[_lhi]

                    same_height = abs(bx_l[_height] - bx_r[_height]) <= 0.5*lbx[_height]

                    l_size = max(bx_l[_width]/max(1,len(bx_l[_tx])), bx_r[_width]/max(1,len(bx_r[_tx])))
                    thres_space = l_size *1 #5

                    nearby = (abs(bx_l[_maxx] - bx_r[_minx]) <= thres_space)

                    overlapping = (bx_l[_maxx] > bx_r[_minx] and (abs(bx_l[_rcen][1] - bx_r[_lcen][1]) <= thres_diff))

                    horizontal_level = (abs(bx_l[_rhi] - bx_r[_lhi]) <= thres_diff) and (abs(bx_l[_rcen][1] - bx_r[_lcen][1]) <= thres_diff)
                    
                    same_line = (horizontal_level and nearby and same_height) #or (overlapping)

                    # Same line
                    if same_line:
                        bx0[_line] = line_id
                        try:
                            idx = line_str[line_id].index(lbx[_tx].lower())
                        except:
                            print(line_str[line_id], lbx[_tx])
                        if lbx_left:
                            line_str[line_id] = line_str[line_id][:idx+len(lbx[0])] + ' ' + bx0[0] + line_str[line_id][idx+len(lbx[0]):]
                        else:
                            line_str[line_id] = line_str[line_id][:idx] +  bx0[0] + ' '  + line_str[line_id][idx:]
                        add_box = True
                        break
            if add_box == False:
                line_id += 1

    line_polys = []
    line_texts = []
    line_areas = []
    line_segs = []

    for i in set(box[_line] for box in box_list):
        current_line_boxes = [box for box in box_list if box[_line]==i]
        com_poly = cv2.convexHull(np.concatenate(([box[_bx] for box in current_line_boxes])))
        line_polys.append(com_poly)
        line_texts.append(line_str[i].strip())
        line_areas.append(sum([polygon_area(box[_bx]) for box in current_line_boxes]))
        line_segs.append([box[_bx] for box in current_line_boxes])

    return line_polys, line_texts, line_areas, line_segs


def lines_to_regions(line_boxes, line_texts, line_areas):
    # create basic attributes
    box_list = []
    for box, txt, area in zip(line_boxes, line_texts, line_areas):
        all_x = [int(coord[0]) for coord in box]
        all_y = [int(coord[1]) for coord in box]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        height = max_y - min_y
        box_list.append([txt, box, area, min_x, max_x, min_y, max_y, height, 0.5*(min_y+max_y), 0, []]) # last element indicates group
    # cluster boxes into paragraph
    _tx, _bx, _ar, _minx, _maxx, _miny, _maxy, _hi, _cen, _region, _ftx = 0,1,2,3,4,5,6,7,8,9,10

    current_region = 1
    while len([box for box in box_list if box[_region]==0]) > 0:
        box_region0 = [box for box in box_list if box[_region]==0] # group0 = non-group
        # new group
        if len([box for box in box_list if box[_region]==current_region]) == 0:
            box_region0[0][_region] = current_region # assign first box to form new group
            box_region0[0][_ftx]= [box_region0[0][_tx]]
        # try to add group
        else:
            current_region_boxes = [box for box in box_list if box[_region]==current_region]
            add_box = False
            for bx0 in box_region0:
                for idx, rbx in enumerate(current_region_boxes):
                    if bx0[_maxy] < rbx[_miny]: #bx0 top
                        top_box = True
                        top, bot = bx0, rbx
                    else: #bx0 bottom
                        top_box = False
                        top, bot = rbx, bx0


                    #logic 2
                    in_vertical_range =False
                    top_w = top[_maxx] - top[_minx]
                    bot_w = bot[_maxx] - bot[_minx]
                    if (top_w) >= (bot_w): # top longer text
                        thres = round(top_w/max(1,len(top[_tx])))
                        if (bot[_minx] >= top[_minx]-thres and bot[_minx] <= top[_maxx]+thres and bot[_maxx] >= top[_minx]-thres and bot[_maxx] <= top[_maxx]+thres):
                            in_vertical_range = True
                    else:
                        thres = round(bot_w/max(1,len(bot[_tx])))
                        if (top[_minx] >= bot[_minx]-thres and top[_minx] <= bot[_maxx]+thres and top[_maxx] >= bot[_minx]-thres and top[_maxx] <= bot[_maxx]+thres ):
                            in_vertical_range = True

                    #logic 1 
                    #in_vertical_range = (bot[_minx] >= top[_minx] and bot[_minx] <= top[_maxx]) or (top[_minx] >= bot[_minx] and top[_minx] <= bot[_maxx])

                    if in_vertical_range:
                        thres = 0.75*(top[_hi] + bot[_hi])  # 0.75*(bot[_hi]+top[_hi])
                        #if bot[_miny] - top[_maxy] <= thres:
                        if (bot[_cen] - top[_cen]) <= thres:
                            bx0[_region] = current_region

                            full_text = rbx[_ftx]
                            rbox_id = full_text.index(rbx[_tx])
                            if top_box:
                                full_text.insert(rbox_id, bx0[_tx])
                            else:
                                full_text.insert(rbox_id+1, bx0[_tx])
                            bx0[_ftx] = full_text #top[_tx] + bot[_tx]
                            rbx[_ftx] = full_text #top[_tx] + bot[_tx]
                            add_box = True
                            break
            # cannot add more box, go to next group
            if add_box==False:
                current_region += 1
    # arrage order in paragraph
    region_polys = []
    region_texts = []
    region_segs = []
    region_areas = []
    #region_ftxts = []
    for i in set(box[_region] for box in box_list):
        current_region_boxes = [box for box in box_list if box[_region]==i]
        #current_region_boxes = [np.array(bx, dtype=int) for bx in current_region_boxes]
        com_poly = cv2.convexHull(np.concatenate(([box[_bx] for box in current_region_boxes])))
        region_polys.append(com_poly)
        region_texts.append( ' '.join(current_region_boxes[0][_ftx]))
        #region_texts.append([box[_tx] for box in current_region_boxes])
        region_segs.append([box[_bx] for box in current_region_boxes])
        #not the actual bounding area but sum of line areas
        region_areas.append(sum([box[_ar] for box in current_region_boxes]))
        #region_ftxts.append(current_region_boxes[0][_ftx])
        #print('+++', current_region_boxes[0][_ftx]) 

    return region_polys, region_texts, region_areas, region_segs, 
