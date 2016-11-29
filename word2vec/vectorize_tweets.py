from gensim.models.doc2vec import LabeledSentence
import os
import json
import gensim.models
#rootdir = '../../../deb/tweets_with_county'
rootdir = "../data/"


class MySentences(object):
	def __init__(self, dirname):
        	self.dirname = dirname
	
	def __iter__(self):
		print 'Reading files'
		for subdir, dirs, files in os.walk(self.dirname):
			for fn in files:
				print os.path.join(subdir,fn)
				for line_ind,line in enumerate(open(os.path.join(subdir,fn))):
					line_data = json.loads(line)
					print 'The tag is - SENT_%s' % (fn+'-'+str(line_ind))
					yield LabeledSentence(words=line_data["text"].split(),tags=['SENT_%s' % (fn+'-'+str(line_ind))])

print 'Program started'
'''
sentences = MySentences(rootdir)
model = gensim.models.Doc2Vec(alpha=0.025, min_alpha=0.025, min_count=1)
model.build_vocab(sentences)
print 'Completed building vocab'
for epoch in range(1):
	model.train(sentences)
	model.alpha -= 0.002
	model.min_alpha = model.alpha

print 'Training done'
model.save('/tmp/my_model.doc2vec')
'''
model_loaded = gensim.models.Doc2Vec.load('/tmp/my_model.doc2vec')
if model_loaded:
	print 'Loaded'
print 'Vector is - ',model_loaded.docvecs["SENT_part-02753-372"],' || Length of vector is - ',len(model_loaded.docvecs)
