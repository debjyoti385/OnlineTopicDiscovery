# Author : vinitha, deb
import numpy as np
import argparse
import gensim.models
from scipy.spatial.distance import cdist

MODEL_LOCATION ='model/tweet_model.doc2vec'

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
    test_file_path = ''
    cluster_means = []
    cluster_counts = []
    doc2vec_model = []
    
    def __init__(self,num_of_clusters,doc2vec_path,test_file_path):
     	print 'In init method'
  	self.num_of_clusters = num_of_clusters
        self.test_file_path = test_file_path
  	self.doc2vec_model = gensim.models.Doc2Vec.load(doc2vec_path)
        vector_len = len(self.doc2vec_model.docvecs[0])
        self.cluster_means = np.random.rand(num_of_clusters,vector_len)
 	self.cluster_counts = np.zeros((num_of_clusters,), dtype=np.int)
    
    def find_closest_mean(self,doc_vector):
	min_index = -1
	min_dist = 0
	for i,mean in enumerate(self.cluster_means):
	    vec_mean_diff = np.linalg.norm(doc_vector-mean)
	    if vec_mean_diff <= min_dist:
		min_index = i
		min_dist = vec_mean_diff 
	return min_index,min_dist

    def compute_cluster_means(self):
	#Iterating over all the vectors to adjust cluster means
    	for element in self.doc2vec_model.docvecs:
	    best_fit,min_dist = self.find_closest_mean(element)
	    self.cluster_counts[best_fit]+=1
	    self.cluster_means[best_fit]+=(min_dist/self.cluster_counts[best_fit])
	print 'Cluster means are = ',self.cluster_means
 
    def classify_test_data(self):
	test_classification = []
	for line in open(self.test_file_path,"r"):
	    words = line.split();
 	    test_vec = self.doc2vec_model.infer_vector(words)
	    best_fit,min_dist = self.find_closest_mean(test_vec)
	    test_classification.append(best_fit)
	return best_fit	   
 
def main(num_of_clusters,test_file_path):
    k_means = Kmeans(num_of_clusters,MODEL_LOCATION,test_file_path)
    k_means.compute_cluster_means()

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
