'''
中文聚类
'''

import numpy as np
import os,time
from preprocess_text import preprocess_text
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import jieba
from sklearn.cluster import KMeans
import pymysql
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances

TFIDF_VOCAB_FILE = "models/tfidffeature.pkl"
TFIDF_TRANSFORMER_FILE = "models/tfidftransformer.pkl"
CLUSTER_FILE = f"models/cluster50_{time.strftime('%Y%m%d%H%I%S')}.pkl"
STOP_WORD_FILE = "stop_word.txt"
jieba.enable_parallel(4)
def load_stop_words():
    stop_words = []
    with open(STOP_WORD_FILE) as f_stop:
        stop_words = f_stop.read().split('\n')
    return stop_words

def segment_txt(txt,stop_words=[]):
    #list,每个语料一行
    return [w for w in jieba.cut(txt) if w not in stop_words]

def get_cluster_corpus():
    corpus = []
    mysql_db = pymysql.connect(host='192.168.0.200', port=3306, user='root', passwd='Wisview!123', db='corpus', charset='utf8mb4')
    mysql_cursor = mysql_db.cursor(cursor=pymysql.cursors.DictCursor)
    corpus = []
    sql = f"select tweet_text from tweet limit 10"
    mysql_cursor.execute(sql)
    tweet = mysql_cursor.fetchone()
    print("load corpus from mysql...")
    lines = 0
    while tweet:
        res_line = preprocess_text(tweet['tweet_text'], DEBUG=False)
        if res_line[0] != 'black_ret':
            corpus.append(res_line[0])
        tweet = mysql_cursor.fetchone()
        lines +=1
        if lines%10000==0:
            print(f'finish {lines}')

    """
    with open("tweets.txt") as f:
        raw = f.readline()
        while raw:
            res_line = preprocess_text(raw, DEBUG=False)
            corpus.append(res_line[0])
            raw = f.readline()
    #list,每个语料一行
    """
    return corpus

if __name__ == '__main__':
    # 加载特征
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(TFIDF_VOCAB_FILE, "rb")))
    # 加载TfidfTransformer
    tfidftransformer = pickle.load(open(TFIDF_TRANSFORMER_FILE, "rb"))
    # 测试用transform，表示测试数据，为list

    test_content=[]
    x_test = ['反差婊羞辱古力娜扎大奶娜扎真的很骚，还这样盯着看，需要你的精液????','零八宪章泽连斯基：俄罗斯占领乌克兰20%领土[url]']
    #x_test = get_cluster_corpus()

    stop_word = load_stop_words()
    print("begin cut words...")
    lines = 0
    for raw in x_test:
        test_content.append(' '.join(segment_txt(raw,stop_word)))
        lines +=1
        if lines % 10000==0:
            print(f"finish {lines}")

    test_tfidf = tfidftransformer.transform(loaded_vec.transform(test_content))
    print('Start predict...')
    #clf = pickle.load(open('./models/seeds/cluster50.pkl', "rb")) 
    clf = pickle.load(open('./models/twitter_spider/cluster50_twitter_spider.pkl', "rb"))
    label = clf.predict(test_tfidf)
  
    tfidf_vect = test_tfidf[0].todense()
    tfidf_vect1 = test_tfidf[1].todense()
    print(euclidean_distances(tfidf_vect,tfidf_vect1))
    #print(tfidf_vect[:100])
    #print(clf.cluster_centers_[0][:100])
    for i in range(len(clf.cluster_centers_)):
        print(i,':',euclidean_distances(clf.cluster_centers_[i].reshape(1,-1),tfidf_vect))
        #print(test_tfidf[0])
    # 每个样本所属的簇
    #i = 0
    #print("save predict result...")
    #print(x_test)
    print(label)
    #with open(f'result_{time.strftime("%Y%m%d%H%I%S")}.txt','w') as fw:
    #    for i in range(len(label)):
    #        fw.write(x_test[i].replace(",",'')+","+str(label)+"\n")




