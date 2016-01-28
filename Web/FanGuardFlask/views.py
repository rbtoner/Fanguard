from flask import render_template
from FanGuardFlask import app
import pandas as pd
import pytumblr
import MySQLdb as mdb

import request
from flask import request

#My shitty model:
from a_Model import ModelIt
import cleaners
import postGather

@app.route('/')
@app.route('/index')
def index():
    return render_template("input.html")

@app.route('/output')
def blogcheck_output():

    #pull 'state' from input field and store it
    name = request.args.get('blog_name')
    
    myconfig = "/Users/ruthtoner/CodingMacros/ProjectInsight/myconfigs.cfg"
    df = postGather.FrontendGetPosts(name,100,myconfig)
    pred = ModelIt("sw",'cut80',df)

    blog_posts = []
    for i in range(0, df.shape[0]):
        bname = df.iloc[i]['blog_name']
        myid = df.iloc[i]['id']
        mytime=df.iloc[i]['date']
        prob=pred[i]
        mytags=df.iloc[i]['taglist']
        mytext=df.iloc[i]['words']
        myurl=df.iloc[i]['short_url']

        blog_posts.append(dict(text=mytext, myid=myid, time=mytime, \
                            prob=pred[i],tags=mytags,author=bname,myurl=myurl))
                           

  #This doesn't work:
  #ip_name=[]
  #ip_name = request.args.get('ip_name')
  #print request.form

    the_result = 0

    return render_template("output.html", blog_name = name, blog_posts = blog_posts, the_result = the_result)
