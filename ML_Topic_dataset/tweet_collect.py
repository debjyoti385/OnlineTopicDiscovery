import tweepy
import os 
import time
import codecs
import sys
from tweepy import StreamListener
from tweepy import Stream

reload(sys)
sys.setdefaultencoding('utf-8')
#g=open("FACup_ids/5_5_2012_14_0.txt",'r');
dirname = "FACup_ids"
consumer_key='W0wAfqirB786tf7QxiKxVmtjt'
consumer_secret='2BNI5fY67S0ewjrDhTBlyDXbWhCE7SNmyg9ppNoE3Jp55wty6o'
access_token='37893525-qOxIXuRpVDLXpv0te3mOVqd4IKFcIU68pkanY6xbG'
access_token_secret='GfD200VrffjbUTScuUTWcgeWgXvBo9q6b8mbwpstMQKLM'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
#wq=g.readlines()
num_calls = 0
tid = list()
output_file = open("output.txt","w+")
for subdir, dirs, files in os.walk(dirname):
    for fn in files:
	print os.path.join(subdir,fn)
	for line in open(os.path.join(subdir,fn)):
	    tid.append(int(line.rstrip('\n')))
	    if len(tid) >= 100:
	  	if num_calls == 800:
		    #after making around 890 calls, stop api calls for 15 minutes
		    time.sleep(900)
		    num_calls = 0
		print 'Making API call, count - ',(num_calls+1)
		stats = api.statuses_lookup(tid)
		#stweet = [s.text for s in stats]
		for tweet in stats:
		    output_file.write("\n"+str(tweet.id)+" * "+str(tweet.text))
		tid = []
		num_calls += 1

output_file.close()
		
'''
for l in wq:
    f=open("output.txt",'a');
    l.rstrip('\n')
    tid=l.split("\t")[0]
    tid=int(tid)
    tid=[tid]   #### Vinitha: you can add upto 100 id's for one call.. bcz every API call is expensive we can call upto 900 request per 15 minutes 
    stats=api.statuses_lookup(tid)
    count += 1
    if count == 10:
        break
    #parenttweet=[s.in_reply_to_status_id for s in stats]
    stweet=[s.text for s in stats]
    #if stweet:
    f.write("\n"+ str(tid) + " "+str(stweet))
    #while parenttweet:
    #    st=api.statuses_lookup(parenttweet)
    #    stw=[x.text for x in st]
    #    if stw:
    #        f.write(" "+str(stw))
    #    parenttweet=[i.in_reply_to_status_id for i in st]
    f.close()
g.close()
'''
