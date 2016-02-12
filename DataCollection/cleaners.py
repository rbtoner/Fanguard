from bs4 import BeautifulSoup  
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def make_text(x,wtitle=True):
    """Concatenate tumblr text fields from post x
    wtitle = include title?
    """
        
    tot_txt = ""

    #Add together various fields based on type:
    if x['type'] == 'text':    
        if wtitle:
            tot_txt = x['title'] + x['body']
        else:
            tot_txt = x['body']
    elif x['type'] == 'photo':
        if wtitle:
            tot_txt = x['title'] + x['caption']
        else:
            x['caption']
    elif x['type'] == 'answer':
        tot_txt = x['question'] + x['answer']
    elif x['type'] == 'chat':
        tot_txt = ""#x['text']

    #Clean out html:
    soup = BeautifulSoup(tot_txt,'html')
        
    return soup.get_text()

def cleaner(x,wtitle=True):
    """Clean text x by removing non-alphanumerics and case, and lemmatizing.
    wtitle = include title?
    """
    
    post_txt=make_text(x,wtitle)

    # Use regular expressions to do a find-and-replace
    letters_only = re.sub("[^a-zA-Z]"," ", post_txt )  # The text to search

    #Lower-bcase"
    lower_case = letters_only.lower()        # Convert to lower case

    #split them up:
    words = lower_case.split(" ")
    str_return=""

    #Lemmatizer!
    wnl = WordNetLemmatizer()

    #Loop over words and lemmatize before re-joining:
    for w in words:
        if (len(w) > 1) and (w not in stopwords.words("english")):
            str_return += wnl.lemmatize(w)
            str_return += " "

    #Encde as ascii:
    return str_return.encode('ascii')

def count(x):
     """Count words in x.
    """
          
    post_txt=make_text(x)

    count = len(re.findall(r'\w+', post_txt))
    return count

def gather_tags(x):
    """Grab tags in x, clean them, and turn them into a list of quoted strings.
    """
        
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
