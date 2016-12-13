from gensim.models.doc2vec import LabeledSentence
import os
import json
import gensim.models
import re
import argparse
#rootdir = '../../../deb/tweets_with_county'
rootdir = "../data/"
MODEL_LOCATION ='model/tweet_model.doc2vec'

text_regex =re.compile('\s*\"text\":\s*\"([^"]+)\"')

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        print 'Reading files'
        for subdir, dirs, files in os.walk(self.dirname):
            for fn in files:
                print os.path.join(subdir,fn)
                for line_ind,line in enumerate(open(os.path.join(subdir,fn))):
                    #line_data = json.loads(line)
                    print line_ind, line
                    text = re.findall(text_regex,line)
                    #print text[0]
                    #print 'The tag is - SENT_%s' % (fn+'-'+str(line_ind))
                    yield LabeledSentence(words=text[0].split(),tags=['SENT_%s' % (fn+'-'+str(line_ind))])



def main(inputdir=rootdir,model=False, modelfile=MODEL_LOCATION):
    print 'Program started'
    if model:
        sentences = MySentences(inputdir)
        model = gensim.models.Doc2Vec(alpha=0.025, min_alpha=0.025, min_count=1)
        model.build_vocab(sentences)
        print 'Completed building vocab'
        for epoch in range(1):
            model.train(sentences)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        
        print 'Training done \nStoring model at location ', modelfile
        model.save(modelfile)

    model_loaded = gensim.models.Doc2Vec.load(modelfile)
    if model_loaded:
        print 'Loaded'
    #print 'Vector is - ',model_loaded.docvecs["SENT_part-02753-372"],' || Length of vector is - ',len(model_loaded.docvecs)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
                            description='Vectorizer for Online Topic Detection')
    parser.add_argument(
                    '-c','--create', help='Train and create model ', default= False, required=False, action='store_true')
    parser.add_argument(
                    '-i','--input', help='input data directory ', default= False, required=True, type=str)
    parser.add_argument(
                    '-o','--output', help='output model file ', default= False, required=True, type=str)
    args = parser.parse_args()
    main(inputdir=args.input,model=args.create, modelfile=args.output)
