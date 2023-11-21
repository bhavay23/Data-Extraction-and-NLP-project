pip install nltk

import os
import re
import glob
import nltk
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('~/work/Input.csv')

print(df.head())

url_ids = df.URL_ID.values
urls = df.URL.values

def extract(id_, url):
    page = urlopen(url)
    page_soup = soup(page, "html.parser")
    title = page_soup.title.string
    texts = ''
    for txt in page_soup.article.find_all('p', text=True):
        texts += (' ' + txt.text)
    texts = texts.strip()
    return title, texts

for id_, url in tqdm(zip(url_ids, urls), total=urls.shape[0]):
    try:
        title, texts = extract(id_, url)
        if os.path.exists('./output_txt/' + str(id_)+'.txt'):
            continue
        with open('./output_txt/' + str(id_)+'.txt', 'w') as f:
            f.write(title)
            f.write('\n')
            f.write(texts)
    except Exception as e:
        print(id_, url, e)
        
txt_files = glob.glob("./output_txt/*.txt")
stop_files = glob.glob("./stopwords/*.txt")
dict_files = glob.glob("./master_dict/*.txt")

stopword_list = []
for file in stop_files:
    df = pd.read_csv(file, sep='|', encoding='latin-1', header=None)
    stopword_list += list(df[0].values)
stopword_list = set([str(x).lower() for x in stopword_list])
print(len(stopword_list))

positive_list = []
negative_list = []
for file in dict_files:
    df = pd.read_csv(file, encoding='latin-1', header=None)
    if 'positive' in file:
        positive_list += list(df[0].values)
    else:
        negative_list += list(df[0].values)

positive_list = set([str(x).lower() for x in positive_list]) 
negative_list = set([str(x).lower() for x in negative_list])
print(len(positive_list))
print(len(negative_list))

def clean_texts(text, stopword_list):
    tokens = word_tokenize(text)
    words = [tok for tok in tokens if tok.lower() not in stopword_list]
    return words


def calculate_score(words, positive_list, negative_list):
    positive_score = 0
    negative_score = 0
    for i in words:
        if i in  positive_list:
            positive_score +=1
        if i in  negative_list:
            negative_score +=1
    
    polarity_score = (positive_score - negative_score)/(positive_score + negative_score + 1e-5)
    subjectivity_score = (positive_score + negative_score)/(len(words) + 1e-5)
    return positive_score,negative_score,polarity_score,subjectivity_score

def readability_analysis(texts):
    sent = texts.split('.')
    words = texts.split(' ')
    avg_sent_len = len(words) / len(sent)
    comp_w = 0
    sum_comp = 0
    for word in words:
        count = count_syllables(word)
        if count > 2:
            comp_w += 1
        sum_comp += count
    
    syll_per_word = sum_comp / len(words)
    perc_comp_word = comp_w / len(words)
    fog = 0.4 * (avg_sent_len + perc_comp_word)
    return syll_per_word, avg_sent_len, comp_w, fog, perc_comp_word
    

def count_syllables(word):
    count = 0
    word = re.sub(r'[^\w\s]','',word)
    word = re.sub(r'_','',word)
    word = word.strip()
    for c in word:
        if c.lower() in ['a', 'e', 'i', 'o', 'u']:
            count += 1
    if word[-2:].lower() in ['es', 'ed']:
        count -= 1
    return count


def count_pronoun(texts):
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(texts)
    return len(pronouns)

def get_avg_word_len(words):
    char_cnt = 0
    for w in words:
        char_cnt += len(w)
    
    return char_cnt / len(words)

def word_count(texts):
    from nltk.corpus import stopwords     
    nltk_words = list(stopwords.words('english'))
    words = clean_texts(texts, nltk_words)
    words = [re.sub(r'[^\w\s]','', re.sub(r'_','',w)).strip() for w in words]
    return len(words), words

out = pd.read_csv('~/work/Output data structure.csv')
ids = out.URL_ID.astype('str')
for file in tqdm(txt_files, total=len(txt_files)):
    with open(file) as f:
        texts = f.read()
    texts = texts.replace('\n', '. ')
    words = clean_texts(texts,stopword_list)
    positive_score,negative_score,polarity_score,subjectivity_score = calculate_score(words, positive_list, negative_list)
    syll_per_word, avg_sent_len, comp_w, fog, perc_comp_word = readability_analysis(texts)
    num_pronoun = count_pronoun(texts)
    count, words = word_count(texts)
    avg_word_len = get_avg_word_len(words)
    idx = np.where(ids == file.split('/')[-1][:-4])[0]
    out.loc[idx, 'POSITIVE SCORE'] = positive_score
    out.loc[idx, 'NEGATIVE SCORE'] = negative_score
    out.loc[idx, 'POLARITY SCORE'] = polarity_score
    out.loc[idx, 'SUBJECTIVITY SCORE'] = subjectivity_score
    out.loc[idx, 'FOG INDEX'] = fog
    out.loc[idx, 'AVG NUMBER OF WORDS PER SENTENCE'] = avg_sent_len
    out.loc[idx, 'PERCENTAGE OF COMPLEX WORDS'] = perc_comp_word
    out.loc[idx, 'AVG SENTENCE LENGTH'] = avg_sent_len
    out.loc[idx, 'SYLLABLE PER WORD'] = syll_per_word
    out.loc[idx, 'COMPLEX WORD COUNT'] = comp_w
    out.loc[idx, 'PERSONAL PRONOUNS'] = num_pronoun
    out.loc[idx, 'AVG WORD LENGTH'] = avg_word_len
    out.loc[idx, 'WORD COUNT'] = count

out.to_csv('./final_submission.csv', index=False)