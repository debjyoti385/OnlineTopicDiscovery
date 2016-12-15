# Author : vinitha, deb
import numpy as np
import argparse
import os
import sys
import random
import re
import time
import gensim.models
from scipy.spatial.distance import cdist

MODEL_LOCATION ='../model/tweet_model.doc2vec'
text_regex =re.compile('\s*\"text\":\s*\"([^"]+)\"')

def load_model():
    model_loaded = gensim.models.Doc2Vec.load(MODEL_LOCATION)
    if model_loaded:
        print 'Loaded model'
    else:
	print 'Unable to load the model at - ',MODEL_LOCATION
    vector_len = len(model_loaded.docvecs[0])
    return model_loaded,vector_len
   
class Kmeans(object):
    num_of_clusters = 0
    test_dir_path = ''
    cluster_means = []
    cluster_counts = []
    doc2vec_model = []
    cluster_cost = []
    cluster_vectors = dict()
    test_cluster_count = []
 
    def __init__(self,num_of_clusters,doc2vec_path,test_dir_path):
     	print 'In init method'
  	self.num_of_clusters = num_of_clusters
        self.test_dir_path = test_dir_path
  	self.doc2vec_model = gensim.models.Doc2Vec.load(doc2vec_path)
        vector_len = len(self.doc2vec_model.docvecs[0])
	self.cluster_means = random.sample(self.doc2vec_model.docvecs,self.num_of_clusters)
        self.cluster_cost = np.zeros((self.num_of_clusters), dtype=np.float)
	self.test_cluster_count = np.zeros((self.num_of_clusters), dtype=np.int)
 	#print 'Initialized mean values - '
	#print self.cluster_means
	self.cluster_counts = np.zeros((num_of_clusters,), dtype=np.int)
    
    def get_point_distance(self,vector, point):
        return np.linalg.norm(vector - point)
    

    def find_closest_mean(self,doc_vector,is_testing=False):
	min_index = -1
	min_dist = sys.maxint
	for i,mean in enumerate(self.cluster_means):
	    vec_mean_diff = np.linalg.norm(doc_vector - mean)
	    if vec_mean_diff < min_dist:
		min_index = i
		min_dist = vec_mean_diff 
	return min_index,min_dist

    def compute_cluster_means(self):
	#Iterating over all the vectors to adjust cluster means
    	for element in self.doc2vec_model.docvecs:
	    best_fit,min_dist = self.find_closest_mean(element,is_testing=False)
	    self.cluster_counts[best_fit]+=1
	    self.cluster_cost[best_fit]+=min_dist
	    diff_vector = np.subtract(self.cluster_means[best_fit],element)
	    self.cluster_means[best_fit]+=(np.true_divide(diff_vector,self.cluster_counts[best_fit]))
	    if best_fit in self.cluster_vectors:
	        self.cluster_vectors[best_fit].append(element)
	    else:
		self.cluster_vectors[best_fit] = [element]

    def compute_cluster_cost(self):
	for i,mean in enumerate(self.cluster_means):
	    for vector in self.cluster_vectors[i]:
   		self.cluster_cost[i]+=self.get_point_distance(vector,mean)
	return self.cluster_cost
		
    def classify_test_data(self):
	test_classification = list()
	for subdir, dirs, files in os.walk(self.test_dir_path):
            for fn in files:
                #print os.path.join(subdir,fn)
                for line in open(os.path.join(subdir,fn)):
		    content = {}
		    text = re.findall(text_regex,line)
		    test_vec = self.doc2vec_model.infer_vector(text[0].split())
		    best_fit,min_dist = self.find_closest_mean(test_vec,is_testing=True)
		    #Incrementing the cluster count for test data
		    self.test_cluster_count[best_fit]+=1
		    content['tweet'] = text[0]
		    content['cluster_id'] = best_fit 
		    test_classification.append(content)
	return test_classification,self.test_cluster_count
 
def main(num_of_clusters,test_dir_path):
    k_means = Kmeans(num_of_clusters,MODEL_LOCATION,test_dir_path)
    k_means.compute_cluster_means()
    cluster_cost = k_means.compute_cluster_cost()
    print 'Cluster costs are - ',cluster_cost
    classification,cluster_count = k_means.classify_test_data()
    print 'Cluster counts for test - ',cluster_count
    
    #for tweet in classification:
    #    print 'Cluster ID is - ',tweet['cluster_id'] 

if __name__ =="__main__":
    parser = argparse.ArgumentParser(
                            description='Algorithm for Sequential K means')
    parser.add_argument(
                    'cluster_count',help='Number of clusters',type=int )
    parser.add_argument(
                    'test_dir_path',help='Path to test data file',type=str)
   
    args = parser.parse_args()
    main(num_of_clusters=args.cluster_count,test_dir_path=args.test_dir_path)
    #main(num_of_clusters=10,test_dir_path='')
