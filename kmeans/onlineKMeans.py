# Author : vinitha, deb
import numpy as np
import sys
import argparse
import gensim.models
import math
#from scipy.spatial.distance import cdist

MODEL_LOCATION ='model/tweet_model.doc2vec'

def load_model():
    model_loaded = gensim.models.Doc2Vec.load(MODEL_LOCATION)
    if model_loaded:
        print 'Loaded model'
    else:
	print 'Unable to load the model at - ',MODEL_LOCATION
    vector_length = len(model_loaded.docvecs[0])
    return model_loaded,vector_length
   
class OnlineKMeans(object):
    """
    Parameters
    ----------
    k_target : int
        Number of clusters to target for OnlineKMeans.
    """
    k_target = 0
    k_actual = -1
    k = 0
    init_delta = 15
    r = 0
    q=[]
    q[r] = 0
    f=[]
    f[r] = 0
    vector_length = 0
    
 
    clusters = {}
    clusters[r] = set()
    cluster_count = {}
    cluster_means = {}
    points =[]   
    size = 0
    
    def __init__(self,k_target,vector_length, init_delta=15):
     	print 'In init method'
        self.k_target = k_target
        if k_target < 15 and k_target > 3:
            init_delta = 3
        self.k = ceil( (k_target - init_delta)/5 ) 
        self.vector_length = vector_length
    
    def point_distance(vector, point):
        return np.linalg.norm(vector - point)

    def cluster_distance(vector, cluster_num):
        points  = self.clusters[cluster_num]
        min_value = sys.maxint
        for p in self.points:
            dist = point_distance(vector, p)
            if dist < min_value:
                min_value = dist
        return min_value
        
    def D(vector):
        min_distances = []
        for i in range(self.k_actual):
            distance  = point_distance(vector, self.cluster_means[i])
            min_distances.append((i,distance))
        min_distances.sort(key=lambda d:d[1])
        min = min_distances[0][1]
        cluster_num = min_distances[0][0]
        for i in range(3):
            temp = cluster_distance(vector, min_distances[i][0])
            if temp < min:
                cluster_num = min_distances[i][0]
                min= temp
        return min, cluster_num
            
    def get_nearest_cluster(vector):
        min= sys.maxsize
        cluster_idx = 1
        for i,p in cluster_means:
            dist = point_distance(vector, p)
            if dist < min:
                cluster_idx = i
                min = dist
        return cluster_idx

    def calculate_w():
        min_distances = []
        length = len(self.points)
        for i in range(length):
            for j in range(i+1,length):
                distance = point_distance(self.points[i],self.points[j])
                min_distance.append(distance)
        min_distance.sort()
        sum = 0
        for i in range(10):
            sum += pow(min_distance[i],2)
        return sum    


    def isTrue(probability):
        if random.uniform(0, 1) > probability:
            return True
        return False


    def add(vector):
        if len(vector) != self.vector_length:
            print "Vector size not compatible, need vector size of " + str(self.vector_length)
            return 

        self.points.append(vector)
        if len(points) < self.k +5:
            self.k_actual += 1
            cluster = clusters.get(self.k_actual,set())
            cluster.add(vector)
            clusters[self.k_actual] = cluster
            cluster_means[self.k_actual] = cluster_means.get(self.k_actual,vector)
            cluster_count[self.k_actual] = cluster_count.get(self.k_actual,0) + 1
            self.w = calculate_w() 
            self.f[0] = self.w
            return 
        else:
            probability = min([pow(D(vector),2)/self.f[self.r],1.0])
            if isTrue(probability):
                self.k_actual += 1
                cluster = clusters.get(self.k_actual,set())
                cluster.add(vector)
                clusters[self.k_actual] = cluster
                cluster_means[self.k_actual] = cluster_means.get(self.k_actual,vector)
                cluster_count[self.k_actual] = cluster_count.get(self.k_actual,0) + 1
                self.q[self.r] += 1
            
                if self.q[self.r] >= self.k:
                    self.r += 1
                    self.q[self.r] = 0
                    self.f[self.r] = 10 * self.f[self.r -1]
            else:
                cluster_idx = get_nearest_cluster(vector)                
                clusters[cluster_idx].add(vector)
                cluster_count[cluster_idx] = cluster_count.get(cluster_idx,0) + 1
            return 
            
        

def main(k_traget,stream):
    online_k_means = OnlineKMeans(k_target=20)
    for vector in stream:
        online_k_means.add(vector)
    print online_k_means.clusters.keys()

if __name__ =="__main__":
    parser = argparse.ArgumentParser(
                            description='Algorithm for Sequential K means')
    parser.add_argument(
                    'cluster_count',help='Number of clusters',type=int )
    parser.add_argument(
                    'test_file_path',help='Path to test data file',type=str)
   
    args = parser.parse_args()
    main(num_of_clusters=args.cluster_count,test_file_path=args.test_file_path)
    #main(num_of_clusters=10,test_file_path='')
