# Author : vinitha, deb
import numpy as np
import argparse
import os
import sys
import random
import re
import time
import gensim.models
import ujson

text_regex =re.compile('\s*\"text\":\s*\"([^"]+)\"')

def load_model():
    model_loaded = gensim.models.Doc2Vec.load(MODEL_LOCATION)
    if model_loaded:
        print 'Loaded model'
    else:
	print 'Unable to load the model at - ',MODEL_LOCATION
    vector_len = len(model_loaded.docvecs[0])
    return model_loaded,vector_len
   
class Kmeans_Ext(object):
    num_of_clusters = 0
    test_dir_path = ''
    cluster_means = []
    cluster_counts = []
    doc2vec_model = []
    cluster_cost = 0
    cluster_vectors = dict()
    test_cluster_count = []
    output_file = ''
    cluster_cost = 0
 
    def __init__(self,num_of_clusters,doc2vec_file,test_dir_path,output_file_path):
     	print 'In init method'
  	self.num_of_clusters = num_of_clusters
        self.test_dir_path = test_dir_path
  	self.doc2vec_model = gensim.models.Doc2Vec.load(doc2vec_file)
        vector_len = len(self.doc2vec_model.docvecs[0])
	self.cluster_means = random.sample(self.doc2vec_model.docvecs,self.num_of_clusters)
	self.test_cluster_count = np.zeros((self.num_of_clusters), dtype=np.int)
	self.cluster_counts = np.zeros((num_of_clusters,), dtype=np.int)
	self.output_file = output_file_path    
	self.compute_cluster_means()

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
	kmeans_model = KMeans(n_clusters=self.num_of_clusters,random_state=0).fit(self.doc2vec_model.docvecs)
	self.cluster_means = kmeans.cluster_centers_
	self.compute_cluster_cost()

    def compute_cluster_cost(self):
	for vector in self.doc2vec_model.docvecs:
	    best_fit,min_dist = find_closest_mean(vector,False)
	    self.cluster_cost+=min_dist

    def print_results(self):
	print 'Number of clusters = ',self.num_of_clusters,' Cost = ',self.cluster_cost

    def classify_test_data(self):
	test_classification = dict()
	for subdir, dirs, files in os.walk(self.test_dir_path):
            for fn in files:
                #print os.path.join(subdir,fn)
                for line in open(os.path.join(subdir,fn)):
		    content = {}
		    text = re.findall(text_regex,line)
		    if not text:
			continue
		    test_vec = self.doc2vec_model.infer_vector(text[0].split())
		    best_fit,min_dist = self.find_closest_mean(test_vec,is_testing=True)
		    #Incrementing the cluster count for test data
		    self.test_cluster_count[best_fit]+=1
		    if best_fit in test_classification:
   			test_classification[best_fit].append(text[0])
		    else:
			test_classification[best_fit] = [text[0]]
	with open(self.output_file,'w+') as output:
            result ={}
            result["name"] ="cluster"
            children =[]
            for key in test_classification:
                temp={}
                temp["name"] = key
                temp["children"]=[ {"name":point, "size":1} for point in test_classification[key]]
                children.append(temp)

            result["children"]=children
            output.write(ujson.dumps(result))

 
def main(num_of_clusters,doc2vec_file,test_dir_path,output_file_path):
    k_means = Kmeans_ext(num_of_clusters,doc2vec_file,test_dir_path,output_file_path)
    k_means.classify_test_data()
    k_means.print_results()
    

if __name__ =="__main__":
    parser = argparse.ArgumentParser(
                            description='Algorithm for K means')
    parser.add_argument(
                    '-k','--k', type=int, help='Target number of clusters ', default= False, required=True)
    parser.add_argument(
                    '-i', '--input',help='Test data directory',required=True, type=str)
    parser.add_argument(
                    '-o', '--output',help='output cluster json file ',default="clusters.json", required=False, type=str)
    parser.add_argument(
                    '-m', '--model',help='Doc2Vec model location',required=False, default="../vectorizer/model/tweet_model.doc2vec", type=str)

    args = parser.parse_args()
    main(num_of_clusters=args.k,doc2vec_file=args.model,test_dir_path=args.input,output_file_path=args.output)
    #main(num_of_clusters=10,test_dir_path='')
