import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy 

class Clustering:

    def __init__(self):
        self.x_coord = []
        self.y_coord = []

    def readInput(self):
        file_path = ('data_points.txt')
        f = open(file_path, 'r')
        points = f.readlines()
        
        self.x_coord = points[0].strip()
        self.y_coord = points[1].strip()

        return self.x_coord, self.y_coord

    def formatInput(self, x_coord, y_coord):
        points = []
        x = np.array([int(x) for x in x_coord.split(',')], dtype = int)
        y = np.array([int(y) for y in y_coord.split(',')], dtype = int)
        l = len(x)

        for i in range(l):
            points.append([int(x[i]), int(y[i])])

        return x, y, points

    def plotPoints(self, x, y, clusters, link_type):
        min_x = min(x) - 2
        max_x = max(x) + 2
        min_y = min(y) - 2
        max_y = max(y) + 2

        fig, axes = plt.subplots(1, 2)
        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])

        axes[0].set_title('Dataset')
        axes[0].scatter(x, y)

        colors = ['r', 'g', 'b', 'c']
        axes[1].set_title('Clusters using ' + link_type + ' link')
        for i, l in enumerate(clusters):
            axes[1].scatter(x[i], y[i], color = colors[l])

        plt.show()

    def link(self, points, link_type):
        clustering = AgglomerativeClustering(n_clusters = 4, linkage = link_type)
        clustering_fit = clustering.fit(points)
        return clustering_fit.labels_, clustering_fit.children_

    def plotDendogram(self, children):
        distance = np.arange(children.shape[0])
        no_of_observations = np.arange(2, children.shape[0]+2)
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
        hierarchy.dendrogram(linkage_matrix)
        plt.title('Dendogram')
        plt.show()
        

if __name__ ==  '__main__':
    ac = Clustering()
    x, y = ac.readInput()
    x_arr, y_arr, points = ac.formatInput(x, y)

    labels, children = ac.link(points, 'single')
    ac.plotPoints(x_arr, y_arr, labels, 'single')
    ac.plotDendogram(children)

    labels, children = ac.link(points, 'average')
    ac.plotPoints(x_arr, y_arr, labels, 'average')
    ac.plotDendogram(children)

    labels, children = ac.link(points, 'complete')
    ac.plotPoints(x_arr, y_arr, labels, 'complete')
    ac.plotDendogram(children)
