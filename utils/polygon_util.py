
def polygon_area(points):
    n = len(points) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area


def get_top_point(polygon):
    try:
        ys = [pt[1] for pt in polygon]
        idx = ys.index(max(ys))
    except:
        print(polygon)
        print(1/0)
    return idx