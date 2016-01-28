from bs4 import BeautifulSoup  
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def make_text(x,wtitle=True):

    print x.keys()
    print x['id'],x['type']
    tot_txt = ""
    
    if x['type'] == 'text':    
        if wtitle:
            tot_txt = x['title'] + x['body']
        else:
            tot_txt = x['body']
    elif x['type'] == 'photo':
        if wtitle:
            tot_txt = x['title'] + x['caption']
        else:
            tot_txt = x['caption']
    elif x['type'] == 'answer':
        tot_txt = x['question'] + x['answer']
    elif x['type'] == 'chat':
        #FIX ME
        tot_txt = ""# x['dialogue']

    print tot_txt
        
    if len(tot_txt)>0:
        soup = BeautifulSoup(tot_txt,'html')
        return soup.get_text()

    return ""

def cleaner(x,wtitle=True):

    post_txt=make_text(x,wtitle)

    # Use regular expressions to do a find-and-replace
    letters_only = re.sub("[^a-zA-Z]"," ", post_txt )  # The text to search
        
    lower_case = letters_only.lower()        # Convert to lower case

    words = lower_case.split(" ")
    str_return=""
    
    wnl = WordNetLemmatizer()
    
    for w in words:
        if (len(w) > 1) and (w not in stopwords.words("english")):
            str_return += wnl.lemmatize(w)
            str_return += " "
    
    return str_return.encode('ascii')

def count(x):
       
    post_txt=make_text(x)

    count = len(re.findall(r'\w+', post_txt))
    return count

def gather_tags(x):
    
    tag_txt = ""
    
    for t in x['tags']:
        t = re.sub("[^a-zA-Z ]","", t )
        t = t.lower()  
        words = t.split(" ")
        str_return=""
        wnl = WordNetLemmatizer()
    
        for w in words:
            if (len(w) > 1) and (w not in stopwords.words("english")):
                str_return += wnl.lemmatize(w)
                str_return += " "   

        tag_txt += '"%s",' % str_return
    return tag_txt.encode('ascii')
