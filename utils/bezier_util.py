import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import numpy as np
import os
import math
from .bezier import Bezier

from scipy.special import comb as n_over_k
Mtk = lambda n, t, k: t**k * (1-t)**(n-k) * n_over_k(n,k)
BezierCoeff = lambda ts: [[Mtk(3,t,k) for k in range(4)] for t in ts]

def train(x, y, ctps, lr):
    x, y = np.array(x), np.array(y)
    ps = np.vstack((x, y)).transpose()
    bezier = Bezier(ps, ctps)
    
    return bezier.control_points_f()

def bezier_fit(x, y):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    data = np.column_stack((x, y))

    # try:
    Pseudoinverse = np.linalg.pinv(BezierCoeff(t)) # (9,4) -> (4,9)
    control_points = Pseudoinverse.dot(data)     # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1,:].flatten().tolist()
    return medi_ctp


def is_close_to_linev2(xs, ys, size, thres = 0.05):
        pts = []
        nor_pixel = int(size**0.5)
        for i in range(len(xs)):
                pts.append(Point([xs[i], ys[i]]))
        import itertools
        # iterate by pairs of points
        slopes = [(second.y-first.y)/(second.x-first.x) if not (second.x-first.x) == 0.0 else math.inf*np.sign((second.y-first.y)) for first, second in zip(pts, pts[1:])]
        st_slope = (ys[-1] - ys[0])/(xs[-1] - xs[0])
        max_dis = ((ys[-1] - ys[0])**2 +(xs[-1] - xs[0])**2)**(0.5)

        diffs = abs(slopes - st_slope)
        score = diffs.sum() * max_dis/nor_pixel

        if score < thres:
                return 0.0
        else:
                return 3.0



def bbx_to_bezier(image_path, bbxs, txts, viz=False):
    img = cv2.imread(image_path)        
    data = []
    polys = []
    beziers = []

    for bbx in bbxs:
        coords = [[co[0], co[1]] for co in bbx]
        coords.append(coords[0])
        poly = Polygon(coords)

        x, y = 0,1 
        if len(bbx) == 4:
            P1, P2 = coords[0], coords[1]
            new_top = ((min(P1[x],P2[x])  + abs(P1[x]-P2[x])/2), (min(P1[y],P2[y]) + abs(P1[y]-P2[y])/2))
            P1, P2 = coords[2], coords[3]
            new_bot = ((min(P1[x],P2[x])  + abs(P1[x]-P2[x])/2), (min(P1[y],P2[y]) + abs(P1[y]-P2[y])/2))
            coords.insert(1,new_top)
            coords.insert(4, new_bot)

        pts = []
        for c in coords[:-1]:
            pts.append(c[0])
            pts.append(c[1])

        data.append(np.array([float(x) for x in pts]))
        polys.append(poly)
    
    for iid, ddata in enumerate(data):
        lh = len(data[iid])
        assert(lh % 4 ==0)
        lhc2 = int(lh/2)
        lhc4 = int(lh/4)

        xcors = [data[iid][i] for i in range(0, len(data[iid]),2)]
        ycors = [data[iid][i+1] for i in range(0, len(data[iid]),2)]

        curve_data_top = data[iid][0:lhc2].reshape(lhc4, 2)
        curve_data_bottom = data[iid][lhc2:].reshape(lhc4, 2)

        left_vertex_x = [curve_data_top[0,0], curve_data_bottom[lhc4-1,0]]
        left_vertex_y = [curve_data_top[0,1], curve_data_bottom[lhc4-1,1]]
        right_vertex_x = [curve_data_top[lhc4-1,0], curve_data_bottom[0,0]]
        right_vertex_y = [curve_data_top[lhc4-1,1], curve_data_bottom[0,1]]

        x_data = curve_data_top[:, 0]
        y_data = curve_data_top[:, 1]

        init_control_points = bezier_fit(x_data, y_data)

        learning_rate = is_close_to_linev2(x_data, y_data, img.size)

        x0, x1, x2, x3, y0, y1, y2, y3 = train(x_data, y_data, init_control_points, learning_rate)
        control_points = np.array([
                [x0,y0],\
                [x1,y1],\
                [x2,y2],\
                [x3,y3]                        
        ])

        x_data_b = curve_data_bottom[:, 0]
        y_data_b = curve_data_bottom[:, 1]

        init_control_points_b = bezier_fit(x_data_b, y_data_b)

        learning_rate = is_close_to_linev2(x_data_b, y_data_b, img.size)

        x0_b, x1_b, x2_b, x3_b, y0_b, y1_b, y2_b, y3_b = train(x_data_b, y_data_b, init_control_points_b, learning_rate)
        control_points_b = np.array([
                [x0_b,y0_b],\
                [x1_b,y1_b],\
                [x2_b,y2_b],\
                [x3_b,y3_b]                        
        ])
 
        t_plot = np.linspace(0, 1, 81)
        Bezier_top = np.array(BezierCoeff(t_plot)).dot(control_points)
        Bezier_bottom = np.array(BezierCoeff(t_plot)).dot(control_points_b)

        if viz:
            plt.plot(Bezier_top[:,0],
                    Bezier_top[:,1],         'g-', label='fit', linewidth=1.0)
            plt.plot(Bezier_bottom[:,0],
                    Bezier_bottom[:,1],         'g-', label='fit', linewidth=1.0)        
            plt.plot(control_points[:,0],
                    control_points[:,1], 'r.:', fillstyle='none', linewidth=1.0)
            plt.plot(control_points_b[:,0],
                    control_points_b[:,1], 'r.:', fillstyle='none', linewidth=1.0)

            plt.plot(left_vertex_x, left_vertex_y, 'g-', linewidth=1.0)
            plt.plot(right_vertex_x, right_vertex_y, 'g-', linewidth=1.0)

        outstr = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}||||{}\n'.format(x0,y0,\
                                                                            round(x1, 2), round(y1, 2),\
                                                                            round(x2, 2), round(y2, 2),\
                                                                            round(x3, 2), round(y3, 2),\
                                                                            round(x0_b, 2), round(y0_b, 2),\
                                                                            round(x1_b, 2), round(y1_b, 2),\
                                                                            round(x2_b, 2), round(y2_b, 2),\
                                                                            round(x3_b, 2), round(y3_b, 2),\
                                                                            txts[iid])

                                                        

        #bez = [[x0,y0],[round(x1, 2), round(y1, 2)],[round(x2, 2), round(y2, 2)],[round(x3, 2), round(y3, 2)],[round(x0_b, 2), round(y0_b, 2)],[round(x1_b, 2), round(y1_b, 2)],[round(x2_b, 2), round(y2_b, 2)],[round(x3_b, 2), round(y3_b, 2)]]
        bez = [(int(x0),int(y0)),(int(x1), int(y1)),(int(x2), int(y2)),(int(x3), int(y3)),(int(x0_b), int(y0_b)),(int(x1_b), int(y1_b)),(int(x2_b), int(y2_b)),(int(x3_b), int(y3_b))]
        #bez = rotational_sort(np.array())
        beziers.append(bez)
    
    if viz:
        plt.imshow(img)
        plt.axis('off')

        if not os.path.isdir('bez_vis'):
                os.mkdir('bez_vis')
        plt.savefig('bez_vis/'+os.path.basename(image_path), bbox_inches='tight',dpi=400)
        plt.clf()
    return beziers
