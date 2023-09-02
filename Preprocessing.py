import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
from nltk.stem import WordNetLemmatizer

def preprocessing(question):
  
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    wo_html = re.sub(html, "", question)
    
  
    wo_url = re.sub(r"https?://\S+|www\.\S+", "", wo_html)
  
    wo_ascii =  re.sub(r'[^\x00-\x7f]',r'', wo_url)

    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    wo_specials =  emoji_pattern.sub(r'', wo_ascii)
   
    wo_punct =  re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', " ", wo_specials)

    
    pattern = re.compile(r'[^\w]|[\d_]')
    
    try: 
        res = re.sub(pattern," ", wo_punct).lower()
    except TypeError:
        return text
    
    res = res.split(" ")
    res = list(filter(lambda x: len(x)>3 , res))
    cleaned_text = " ".join(res)

    # opening the file in read mode
    my_file = open("stopwords.txt", "r")
    # reading the file
    stop_words = my_file.read()
    
    try:
        res = word_tokenize(cleaned_text, language='english')
    except TypeError:
        return text
    
    token_output = [token for token in res if token not in stop_words]
    
    res = nltk.pos_tag(token_output)
    
    res = [token[0] for token in res if token[1] in ['NN']]

    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    
    for token in res:
        lemmatized.append(lemmatizer.lemmatize(token))
        
    return [" ".join(lemmatized)]