# Author : vinitha, deb
import ujson
import numpy as np
import re
import sys,os
import argparse
import gensim.models
import random
from math import ceil, pow
#from scipy.spatial.distance import cdist

MODEL_LOCATION ='model/tweet_model.doc2vec'
TWEET_REGEX =re.compile('\s*\"text\":\s*\"([^"]+)\"')

def load_model():
    model_loaded = gensim.models.Doc2Vec.load(MODEL_LOCATION)
    if model_loaded:
        print 'Loaded model'
    else:
        print 'Unable to load the model at - ',MODEL_LOCATION
        exit()
    vector_length = len(model_loaded.docvecs[0])
    return model_loaded,vector_length


class TweetStream(object):
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        #print 'Reading files'
        for subdir, dirs, files in os.walk(self.dirname):
            for fn in files:
                print os.path.join(subdir,fn)
                for line_ind,line in enumerate(open(os.path.join(subdir,fn))):
                    text = re.findall(TWEET_REGEX,line)
                    yield text[0]

   
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
    q.insert(r,0)
    f=[]
    f.insert(r,1)
    vector_length = 0
    
 
    clusters = {}
    clusters_text = {}
    clusters[r] = []
    cluster_count = {}
    cluster_means = {}
    points =[]   
    size = 0
    norm=2
    
    def __init__(self,k_target,vector_length, norm=2, init_delta=15):
     	print 'Initialize Online K Means with k ', str(k_target) , " and distance norm ", str(norm)
        self.k_target = k_target
        self.norm = norm
        if k_target < 15 and k_target > 3:
            init_delta = 3
        self.k = ceil( (k_target - init_delta)/5 ) 
        self.vector_length = vector_length
    
    def point_distance(self,vector, point):
        if self.norm != 2:
            return np.linalg.norm(vector-point,ord=self.norm)
        return np.linalg.norm(vector - point)

    def cluster_distance(self,vector, cluster_num):
        points  = self.clusters[cluster_num]
        min_value = sys.maxint
        for p in points:
            dist = self.point_distance(vector, p)
            if dist < min_value:
                min_value = dist
        return min_value
        
    def D(self,vector):
        min_distances = []
        for i in range(self.k_actual):
            distance  = self.point_distance(vector, self.cluster_means[i])
            min_distances.append((i,distance))
        min_distances.sort(key=lambda d:d[1])
        min_value = min_distances[0][1]
        cluster_num = min_distances[0][0]
        #print min_distances 
        for i in range(min(len(min_distances),3)):
            temp = self.cluster_distance(vector, min_distances[i][0])
            if temp < min_value:
                cluster_num = min_distances[i][0]
                min_value= temp
        #print min_value
        return min_value, cluster_num
            
    def get_nearest_cluster(self,vector):
        min= sys.maxsize
        cluster_idx = 1
        for key in self.cluster_means.keys():
            dist = self.point_distance(vector, self.cluster_means[key])
            if dist < min:
                cluster_idx = key
                min = dist
        return cluster_idx

    def calculate_w(self):
        min_distances = []
        length = len(self.points)
        for i in range(length):
            for j in range(i+1,length):
                distance = self.point_distance(self.points[i],self.points[j])
                min_distances.append(distance)
        min_distances.sort()
        sum = 0
        for i in range(min(len(min_distances),10)):
            sum += pow(min_distances[i],2)
        return sum /2   


    def isTrue(self,probability):
        if random.uniform(0, 1) < probability:
            return True
        return False


    def add(self,vector, text=""):
        if len(vector) != self.vector_length:
            print "Vector size not compatible, need vector size of " + str(self.vector_length)
            return 

        self.points.append(vector)
        if len(self.points) < self.k +5:
            self.k_actual += 1
            cluster = self.clusters.get(self.k_actual,[])
            cluster.append(vector)
            self.clusters[self.k_actual] = cluster
            cluster_text = self.clusters_text.get(self.k_actual,[])
            cluster_text.append(text)
            self.clusters_text[self.k_actual] = cluster_text
            self.cluster_means[self.k_actual] = self.cluster_means.get(self.k_actual,vector)
            self.cluster_count[self.k_actual] = self.cluster_count.get(self.k_actual,0) + 1
            self.w = self.calculate_w() 
            self.r = 0
            self.f[self.r] = self.w
            return 
        else:
            distance, cluster_num =  self.D(vector)
            print distance
            #print self.f[self.r]
            probability = min(pow(distance,2)/self.f[self.r],1.0)
            print probability
            if self.isTrue(probability):
                self.k_actual += 1
                cluster = self.clusters.get(self.k_actual,[])
                cluster.append(vector)
                self.clusters[self.k_actual] = cluster
                cluster_text = self.clusters_text.get(self.k_actual,[])
                cluster_text.append(text)
                self.clusters_text[self.k_actual] = cluster_text
                self.cluster_means[self.k_actual] = self.cluster_means.get(self.k_actual,vector)
                self.cluster_count[self.k_actual] = self.cluster_count.get(self.k_actual,0) + 1
                self.q[self.r] += 1
            
                if self.q[self.r] >= self.k:
                    self.r += 1
                    self.q.insert(self.r, 0)
                    self.f.insert(self.r,  1.05* self.f[self.r -1])
            else:
                cluster_idx = self.get_nearest_cluster(vector)                
                self.clusters[cluster_idx].append(vector)
                self.clusters_text[cluster_idx].append(text)
                self.cluster_count[cluster_idx] = self.cluster_count.get(cluster_idx,0) + 1
            return 
            
        
def extract_tweet(line):
    text = re.findall(TWEET_REGEX,line)
    return text[0]


def main(k_target,datadir, model_path, norm=2, outputfile="output.json"):
    tweets = TweetStream(datadir)
    vectorizer = gensim.models.Doc2Vec.load(model_path)
    vector_length = len(vectorizer.docvecs[0])

    online_k_means = OnlineKMeans(k_target=20, vector_length=vector_length, norm=norm)
    count  = 0
    for tweet in tweets: 
        count +=1
        #extract text from stream

        #print tweet
       
        # convert stream to vector
        words = tweet.split();
        
        vector = vectorizer.infer_vector(words)

        online_k_means.add(20*vector,text=tweet)
        print count
        if count % 500 ==0:
            print "PROCESSED "+ str(count) + " TWEETS"
    with open(outputfile,'w') as output:
        result ={}
        result["name"] ="cluster"
        children =[]
        for key in online_k_means.clusters_text:
            temp={}
            temp["name"] = key
            temp["children"]=[ {"name":point, "size":1} for point in online_k_means.clusters_text[key]]
            children.append(temp)

        result["children"]=children
        output.write(ujson.dumps(result))

if __name__ =="__main__":
    parser = argparse.ArgumentParser(
                            description='Algorithm for Online K means')
    parser.add_argument(
                    '-k','--k', type=int, help='Target number of clusters ', default= False, required=True)
    parser.add_argument(
                    '-i', '--input',help='Input data directory',required=True, type=str)
    parser.add_argument(
                    '-o', '--output',help='output cluster json file ',default="clusters.json", required=False, type=str)
    parser.add_argument(
                    '-m', '--model',help='Doc2Vec model location',required=False, default="../vectorizer/model/tweet_model.doc2vec", type=str)
    parser.add_argument(
                    '-l','--l', type=int, help='Distance Norm ', default=2, required=False)
   
    args = parser.parse_args()
    main(args.k, args.input, args.model, norm=args.l, outputfile=args.output)
