#coding:utf-8
import os
from flask import Flask, request, jsonify, abort, Response
import multiprocessing as mp
import threading
import time
import torch
import logging
from transformers import BertModel,BertConfig,BertTokenizer

from utils.mo_utils import *
import numpy as np

tokenizer = None
model = None
model_dir = u'./cluster_bert/model'
device=torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

def init(n):
    # RGB model
    global tokenizer,model,device
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir, output_hidden_states=True, return_dict=True).to(device)
    #tokenizer = BertTokenizer.from_pretrained(model_dir)
    #config =BertConfig.from_pretrained(model_dir)
    #model = BertModel.from_pretrained(model_dir,output_hidden_states=True, return_dict=True).to(device)
    # model.to(device)

def run(vedio_list):
    global tokenizer,model,device
    tokenized_text = tokenizer(vedio_list, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokenized_text)
        sentence_embedding = outputs.pooler_output
        sentence_embedding = sentence_embedding.tolist()

    clustering_model = load_pickle("./cluster_model/clustering_model50.pkl")
    cluster_assignment = clustering_model.labels_
    sentence_embedding = np.array(sentence_embedding)
    predict_res = clustering_model.predict(sentence_embedding)
    # print(clustering_model.predict(corpus_embeddings[:10]))
    # 每个样本所属的簇
    # label = []  # 存储1000个类标 4个类
    # # print(clf.labels_)
    # i = 0
    # print("save cluster result...")
    # clustered_sentences = [[] for i in range(num_clusters)]
    # # with open(f'./output/result_{time.strftime("%Y%m%d%H%I%S")}.txt', 'w') as fw:
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     clustered_sentences[cluster_id].append(corpus[sentence_id])

    return predict_res

PARALLEL_CNT = 3

app = Flask(__name__)
#sem1 = threading.Semaphore(PARALLEL_CNT)
#init_count_down = PARALLEL_CNT
init_count_down_lock = threading.RLock()

def init_fn(n):
    init(n)
    return True

def init_cb(v):
    logging.error('rel')
    global init_count_down
    init_count_down_lock.acquire()
    init_count_down -= len(v)
    init_count_down_lock.release()

def init_ecb(err):
    print("init_ecb", err, os.getpid())


#work_pool = mp.Pool(processes=PARALLEL_CNT)
#work_pool.map_async(init_fn, range(PARALLEL_CNT),callback=init_cb) # error_callback=init_ecb
#n = 0
#n_lock = threading.RLock()
#tl = threading.local()

@app.route('/predict', methods=['post'])
def predict():
    #global init_count_down,model
    #if init_count_down > 0:
    #    return jsonify({"status": 2, "msg": 'Service is booting...' + time.asctime()})
    #if not sem1.acquire(blocking=False):
    #    return jsonify({"status": 1, "msg": 'Busy...' + time.asctime()})

    #res = dict()
    #res['result'] = []
    #if request.method != 'POST':
    #    res['msg'] = "request method should be Post"
    #    res['status'] = 400
    #    return jsonify(res)
   
    #n_lock.acquire()
    #global n
    #n += 1
    #tl.n = n
    #n_lock.release()

    try:
        video_list = request.get_json()
        print(video_list)
        #res['status'] = 200
        #state, result = work_pool.apply(run, vedio_list['str'])
        #sem1.release()
        if 1:#try:
            ret = run(video_list['str'])
        #except:
        #    print("exp")
        #if state != 0:
        #    jsonify({"status":401, "msg":"the path of image is empty"})
        #else:
        print(ret)
        return jsonify({"logits":ret})
    except ValueError:
        return jsonify({"status":400, "msg":"请求失败"})

if __name__ == "__main__":
    init(1)
    app.run(host="10.96.130.66", port=7099, debug=False, threaded=True, processes=1)
