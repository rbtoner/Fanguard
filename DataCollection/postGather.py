from __future__ import division
import pandas as pd
import numpy as np
import MySQLdb as mdb
import sys
import time
import ConfigParser

#SQL:
from sqlalchemy import create_engine, MetaData, TEXT, Integer, Table, Column, ForeignKey, Float
from sqlalchemy import create_engine

#API:
import pytumblr

#Personal package:
import cleaners

#Generator to grab posts from a series of hashtags:
def PostGenerator(client, tag, timelist = [],rep = 1):
    """Generator to get posts from Tumblr API.
    client = tumblr client
    tag = tag to search for
    timelist = times to search at
    rep = repetition rate, if using timelist
    """
    
    #Grab 20 or less:
    chunk=20
    #Just a counter:
    i=0
    j=0
    #For now, start at current time:
    last_stamp = time.time()

    #Total time measured:
    elapsed=0

    #For as long as we need to:
    while True:

        #If we're using a timelist, and it's time for a new timestamp:
        if (i%rep==0) and (len(timelist)>0):

            #We've run past the stamp!
            if (last_stamp < timelist[j]):
                print "WARNING: %d < %d; will later drop duplicates" \
                    % (last_stamp,timelist[j])

            #New last stamp:
            new_before = timelist[j] - 1
            j += 1 #Increment

        else:
            new_before = last_stamp #New last stamp

        #Print out some progress every 20 calls:
        if (i%20)==0:
            print "%d..." % i,

        #Only more recent than start of Jan, 2013:
        if (new_before < 1356998400):
            print "Past date"
            yield None, 0, 0
        
        #Posts with tag, starting at last_stamp:
        #print "Looking before",new_before
        posts = client.tagged(tag, limit=chunk, before=new_before)
        #print i,j,last_stamp
        
        #Didn't get anything
        if not posts:
            print "Out of posts"
            yield None, 0, 0
        else:
            elapsed = posts[0]['timestamp'] - posts[len(posts)-1]['timestamp']
            if elapsed < 0:
                print "WARNING: final %d < initial %d" % (posts[len(posts)-1]['timestamp'], posts[0]['timestamp'])
            #print "First post at %d, last post at %d" % (posts[0]['timestamp'],posts[len(posts)-1]['timestamp'])
            #Reset the time stamp to that of the oldest:
            last_stamp = posts[len(posts)-1]['timestamp']
            #YIELD
            yield posts,last_stamp,[elapsed/len(posts)]*len(posts)

        #Counters:
        i += 1
        

def GeneratePosts(tag,dbname,reprate,myconfig):
    """Generate posts for a given tag and insert them into database
    tag = tag to search for
    dbname = database name stub
    reprate = reprate for sampling "all" tag
    myconfig = MySQL configurations
    """

    #Get MySQL configurations:
    config = ConfigParser.RawConfigParser()
    config.read(myconfig) 

    #MySQL info:
    db_username = config.get('DB', 'username')
    db_pwd = config.get('DB', 'pwd')

    #Tumblr Key Info:
    consumer_key = config.get('tcred', 'consumer_key')
    consumer_secret = config.get('tcred', 'consumer_secret')
    oauth_token = config.get('tcred', 'oauth_token')
    oauth_secret = config.get('tcred', 'oauth_secret')

    #Get the client:
    client = pytumblr.TumblrRestClient(consumer_key,consumer_secret, \
        oauth_token, oauth_secret)
    
    #Lists to hold all the posts:
    all_posts = []
    spo_posts = []
    
    #Generator to grab posts from spoiler tag:
    spo_gen = PostGenerator(client,"%s-spoilers" % tag)

    time_list = [] #List of timestamp to match with regular

    dtime_spo = [] #weights (if needed)

    #First, grab the total posts (up to 1000 worth):
    for _ in range(0,1000):
        
        time.sleep(0.1) #Don't hit the API rate limit

        #Grab posts, plus stamp and elapsed information
        posts,ts,te = spo_gen.next()

        #When we're out of posts:
        if not posts:
            break
        
        #Append list of posts:
        spo_posts += posts

        #Time information:
        time_list.append(ts)
        dtime_spo += te

    #print time_list
        
    #How much time is represented by the API call?
    print "Total elapsed time (spoilers) =", \
      np.asarray(dtime_spo, dtype=float).sum()

    #Generator to grab posts from regular tag:
    all_gen = PostGenerator(client,tag,time_list,reprate)

    #Timelist and stamps:
    dtime_all=[] 
    stamps=[] #List of timestamps (for debugging)

    #First, grab the total posts (reprate * len(spoilers)):
    for _ in range(0,reprate*len(time_list)):
        
        time.sleep(0.1) #Don't hit the API rate limit

        #Grab posts, plus stamp and elapsed information
        posts,ts,te = all_gen.next()
        #print ts
        #When we're out of posts:
        #if not posts:
        #    break

        if posts:
            #Append list of posts:
            all_posts += posts
            #Time info:
            stamps.append(ts)
            dtime_all += te
        else:
            print "(GAP)"

    
    print "Total elapsed time (all posts) =", \
      np.asarray(dtime_all, dtype=float).sum()

    df_all = pd.DataFrame(all_posts)
    df_spo = pd.DataFrame(spo_posts)
    
    df_all = df_all.replace(np.nan,' ', regex=True)
    df_spo = df_spo.replace(np.nan,' ', regex=True)

    dsql_all = df_all[['id']].copy()
    dsql_spo = df_spo[['id']].copy()

    print "Cleaning Spoiler text..."
    dsql_spo.loc[:,'alltext']= df_spo.apply(cleaners.cleaner,axis=1)
    print "Cleaning Regular text..."
    dsql_all.loc[:,'alltext'] = df_all.apply(cleaners.cleaner,axis=1)
    print "Calculating counts.."
    dsql_spo.loc[:,'count']= df_spo.apply(cleaners.count,axis=1)
    dsql_all.loc[:,'count'] = df_all.apply(cleaners.count,axis=1)
    print "Cleaning Spoiler tags..."
    dsql_spo.loc[:,'taglist']= df_spo.apply(cleaners.gather_tags,axis=1)
    print "Cleaning Regular tags..."
    dsql_all.loc[:,'taglist'] = df_all.apply(cleaners.gather_tags,axis=1)
    print "Copying over other stuff..."
    dsql_spo.loc[:,'blog_name']= df_spo['blog_name'].astype(str)
    dsql_all.loc[:,'blog_name'] = df_all['blog_name'].astype(str)
    dsql_spo.loc[:,'timestamp']= df_spo['timestamp']
    dsql_all.loc[:,'timestamp'] = df_all['timestamp']
    dsql_spo.loc[:,'note_count']= df_spo['note_count']
    dsql_all.loc[:,'note_count'] = df_all['note_count']
    dsql_spo.loc[:,'dtime'] = np.asarray(dtime_spo)
    dsql_all.loc[:,'dtime'] = np.asarray(dtime_all)
    
    print "Done generating trees."

    #Remove duplicates:
    print "Sizes before duplicate dropping:", dsql_spo.shape,dsql_all.shape
    dsql_spo = dsql_spo.drop_duplicates()
    dsql_all = dsql_all.drop_duplicates()
    print "Sizes after duplicate dropping:", dsql_spo.shape,dsql_all.shape
    
    #Connection to DB:
    engine = create_engine("mysql+mysqldb://%s:%s@localhost/InsightData" % (db_username,db_pwd))

    #Upload Spoilers to MySQL:
    print "Starting Spoiler MySQL upload..."
    meta = MetaData(bind=engine)
    spoiler_table = 'tumblr_%s_spoil' % dbname
    ## Spoiler Table###
    table_tumblr_spoiler = Table(spoiler_table, meta,
        Column('id', Integer, primary_key=True, autoincrement=False),
        Column('timestamp', Integer, nullable=False),
        Column('note_count', Integer, nullable=True),  
        Column('count', Integer, nullable=True),                     
        Column('alltext', TEXT, nullable=True),
        Column('taglist', TEXT, nullable=True),
        Column('blog_name', TEXT, nullable=True),
        Column('dtime', Float, nullable=False)
    )
    print "Done..."   
    meta.create_all(engine)
    dsql_spo.to_sql(spoiler_table,engine,flavor='mysql', \
                    if_exists='replace',index=True)

                    
    #Upload Spoilers to MySQL:
    print "Starting Regular MySQL upload..."
    meta = MetaData(bind=engine)
    regular_table = 'tumblr_%s_regular' % dbname
    ## Tumble Table###
    table_tumblr_regular = Table(regular_table, meta,
        Column('id', Integer, primary_key=True, autoincrement=False),
        Column('timestamp', Integer, nullable=False),
        Column('note_count', Integer, nullable=True),  
        Column('count', Integer, nullable=True),                     
        Column('alltext', TEXT, nullable=True),
        Column('taglist', TEXT, nullable=True),
        Column('blog_name', TEXT, nullable=True),
        Column('dtime', Float, nullable=False)
    )
    meta.create_all(engine)
    dsql_all.to_sql(regular_table,engine,flavor='mysql', \
                    if_exists='replace',index=True)
    print "Done..."
    print "FINISHED"

def FrontendGetPosts(tag,npost,myconfig):

    config = ConfigParser.RawConfigParser()
    config.read(myconfig) 

    #MySQL info:
    db_username = config.get('DB', 'username')
    db_pwd = config.get('DB', 'pwd')

    consumer_key = config.get('tcred', 'consumer_key')
    consumer_secret = config.get('tcred', 'consumer_secret')
    oauth_token = config.get('tcred', 'oauth_token')
    oauth_secret = config.get('tcred', 'oauth_secret')
    
    client = pytumblr.TumblrRestClient(consumer_key,consumer_secret, \
        oauth_token, oauth_secret)
    
    #Lists to hold all the posts:
    all_posts = []
    
    #Generator to grab posts from spoiler tag:
    post_gen = PostGenerator(client,tag)
    
    post_lim = int(npost/20)
    
    #First, grab the total posts (up to 1000 worth):
    for _ in range(0,post_lim):

        time.sleep(0.1) #Don't hit the API rate limit
        
        #Grab posts, plus stamp and elapsed information
        posts,ts,te = post_gen.next()

        #When we're out of posts:
        if not posts:
            break
        
        #Append list of posts:
        all_posts += posts

    df_all = pd.DataFrame(all_posts)
    df_all = df_all.replace(np.nan,' ', regex=True)

    df = df_all[['id']].copy()
    df.loc[:,'words'] = df_all.apply(cleaners.cleaner,axis=1)
    df.loc[:,'wcount'] = df_all.apply(cleaners.count,axis=1)
    df.loc[:,'taglist'] = df_all.apply(cleaners.gather_tags,axis=1)
    df.loc[:,'blog_name'] = df_all['blog_name'].astype(str)
    df.loc[:,'date'] = df_all['date']
    df.loc[:,'note_count'] = df_all['note_count']
    df.loc[:,'short_url'] = df_all['short_url']

    return df
