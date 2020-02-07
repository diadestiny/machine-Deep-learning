import math
# 最邻近规则分类算法K-Nearest Neighbor


def ComputeEuclideanDistance(x1, y1, x2, y2):
    d = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    return d


# 电影(打斗次数，接吻次数):(18,90)待归类点
d_ag = ComputeEuclideanDistance(3, 104, 18, 90)
d_bg = ComputeEuclideanDistance(2, 100, 18, 90)
d_cg = ComputeEuclideanDistance(1, 81, 18, 90)
d_dg = ComputeEuclideanDistance(101, 10, 18, 90)
d_eg = ComputeEuclideanDistance(99, 5, 18, 90)
d_fg = ComputeEuclideanDistance(98, 2, 18, 90)

print(d_ag, d_bg, d_cg, d_dg, d_eg, d_fg)
# 得到距离就近的前k个已知点，归到对应的一类(少数服从多数)