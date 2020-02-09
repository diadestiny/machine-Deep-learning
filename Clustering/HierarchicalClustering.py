from numpy import *


class cluster_node:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None, count=1):
        self.left = left
        self.right = right
        self.vec = vec
        self.distance = distance
        self.id = id
        self.count = count


def L2dist(v1, v2):
    return sqrt(sum(v1 - v2) ** 2)


def L1dist(v1, v2):
    return sum(abs(v1 - v2))

# 参数可以传函数
def hcluster(features, distance=L2dist):
    distances = {}
    currentclustid = -1

    # 每个点自成一个类别
    clust = [cluster_node(array(features[i], id=i) for i in range(len(features)))]

    while len(clust) > 1:
        lowstpiar = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                if d < closest:
                    closest = d
                    lowstpiar = (i, j)
        mergeve = [(clust[lowstpiar[0]].vec[i] + clust[lowstpiar[1]].vec[i]) / 2.0 for i in
                   range(len(clust[lowstpiar[1]].vec))]
        newcluster = cluster_node(array(mergeve), left=clust[lowstpiar[0]], right=clust[lowstpiar[1]], distance=closest,
                                  id=currentclustid)
        currentclustid -= 1
        del clust[lowstpiar[1]]
        del clust[lowstpiar[0]]
        clust.append(newcluster)
    return clust[0]


def extract_clusters(clust, dist):
    clusters = {}
    if clust.distance < dist:
        return [clust]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = extract_clusters(clust.left, dist=dist)
        if clust.right != None:
            cr = extract_clusters(clust.right, dist=dist)
        return cl + cr


def get_cluster_element(clust):
    if clust.id >= 0:
        return [clust.id]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = get_cluster_element(clust.left)
        if clust.right != None:
            cr = get_cluster_element(clust.right)
        return cl + cr


def printclust(clust, labels=None, n=0):
    for i in range(n): print(' ')
    if clust.id < 0:
        print('-')
    else:
        if labels == None:
            print(clust.id)
        else:
            print(labels[clust.id])

    if clust.left != None: printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None: printclust(clust.right, labels=labels, n=n + 1)


def getheight(clust):
    if clust.left == None and clust.right == None: return 1
    return getheight(clust.left) + getheight(clust.right)


def getdepth(clust):
    if clust.left == None and clust.right == None: return 0
    return max(getheight(clust.left), getheight(clust.right)) + clust.distance
