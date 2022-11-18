import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import pandas as pd
import numpy as np


import torch
from transformers import BertModel, BertConfig, BertTokenizer
import multiprocessing as mp
from sklearn.cluster import KMeans

from tqdm import tqdm

from utils.mo_utils import *
from utils.preprocess_text import *
model_dir = u'./code/cluster_bert/model'


def data_process(chunk_list, n):
    num = n % 4
    device = torch.device("cuda:{}".format(num) if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    config = BertConfig.from_pretrained(model_dir)
    bert = BertModel.from_pretrained(model_dir, output_hidden_states=True, return_dict=True).to(device)
    text_list = []
    result_list = []
    tmp_list = []
    print(len(chunk_list))
    for i in tqdm(range(len(chunk_list))):
        this_chunk_len = len(chunk_list[i])
        # if len(chunk_list[i]) <= 16 or len(chunk_list[i]) >= 127:
        #     continue
        text = chunk_list[i].replace(' ', '')
        tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = bert(**tokenized_text)
        sentence_embedding = outputs.pooler_output
        tmp_list = []
        tmp_list.append(sentence_embedding.tolist())
        text_list.append(text)
        result_list.append(tmp_list)
    df_all = pd.DataFrame(text_list, columns=['text'])
    df_all = pd.concat([df_all, pd.DataFrame(result_list, columns=['embeddings'])], axis=1)
    return df_all

def get_normal_corpus(off_mod=True):
    data_list_file = "./data/data_list.pkl"
    if off_mod == True:
        data_list = load_pickle(data_list_file)
    else:
        increment_time  = 0
        this_content_time_name = "tt_content_dict_pp" + str(increment_time) + ".pkl"
        tt_content_dict = load_pickle("./output/" + this_content_time_name)
        #
        normal_data_list =  []
        for  i, (ck, cv) in enumerate(tt_content_dict.items()):
            normal_data_list.append(cv[4][0])

        tt_mg_post_name = "tt_mg_post_dict1008.pkl"
        tt_mg_comment_name = "tt_mg_comment_dict0929.pkl"

        tt_mg_post_dict = load_pickle("./tmp/" + tt_mg_post_name)
        tt_mg_comment_dict = load_pickle("./tmp/" + tt_mg_comment_name)

        kw_mg_data_list = []
        for i, (pk, pv) in enumerate(tt_mg_post_dict.items()):
            this_text_raw = pv[0]
            this_text = filter_illegal_char(this_text_raw)
            filtered_content = preprocess_text(this_text, max_sentence_length=255, min_sentence_length=6)

            if len(filtered_content) != 1:
                # print("filter error:", this_id, filtered_content)
                continue

            if len(filtered_content[0]) < 6 or len(filtered_content[0]) >= 255:
                # print("filter so short error:", len(filtered_content[0]), this_id, filtered_content)
                continue

            kw_mg_data_list.append(filtered_content)

        for i, (ck, cv) in enumerate(tt_mg_comment_dict.items()):
            this_text_raw = cv[0]
            this_text = filter_illegal_char(this_text_raw)
            filtered_content = preprocess_text(this_text, max_sentence_length=255, min_sentence_length=6)

            if len(filtered_content) != 1:
                # print("filter error:", this_id, filtered_content)
                continue

            if len(filtered_content[0]) < 6 or len(filtered_content[0]) >= 255:
                # print("filter so short error:", len(filtered_content[0]), this_id, filtered_content)
                continue
            kw_mg_data_list.append(filtered_content)

        data_list = normal_data_list + kw_mg_data_list
        generate_pickle("./data/data_list.pkl", data_list)

    return data_list

def get_infer_samples(off_mod=True):
    data_list_file = "./data/data_list.pkl"
    if off_mod == True:
        data_list = load_pickle(data_list_file)
    else:
        increment_time  = 0
        # this_content_time_name = "tt_content_dict_pp" + str(increment_time) + ".pkl"
        this_content_time_name = "tt_content_pp_dict_" + str(increment_time) + ".pkl"
        tt_content_dict = load_pickle("./output/" + this_content_time_name)
        #
        normal_data_list =  []
        for  i, (ck, cv) in enumerate(tt_content_dict.items()):
            normal_data_list.append(cv[4][0])

        data_list = normal_data_list
        generate_pickle("./data/infer_data_list.pkl", data_list)

    return data_list

def get_text_embedding(data_list, off_mod=True):
    # df_all_file = "./data/df_all.pkl"
    # corpus_embeddings_file = "./data/corpus_embeddings.pkl"
    # corpus_file = "./data/corpus.pkl"

    df_all_file = "./data/df_all_total.pkl"
    corpus_embeddings_file = "./data/corpus_embeddings_total.pkl"
    corpus_file = "./data/corpus_total.pkl"


    if off_mod == True:
        df_all = load_pickle(df_all_file)
        corpus_embeddings = load_pickle(corpus_embeddings_file)
        corpus = load_pickle(corpus_file)
    else:
        results = []
        num_thread = 12
        # pool = mp.Pool(processes=num_thread)
        ctx = torch.multiprocessing.get_context("spawn")
        pool = ctx.Pool(num_thread)
        num_chunk = int(len(data_list) / num_thread)
        chunk_list = [data_list[i * num_chunk:(i + 1) * num_chunk] for i in range(int(len(data_list) / num_chunk) + 1) if
                      data_list[i * num_chunk:(i + 1) * num_chunk]]
        for i in range(0, num_thread):
            result = pool.apply_async(data_process, args=(chunk_list[i], i,))
            results.append(result)
        pool.close()
        pool.join()

        df_all = None
        for i in range(len(results)):
            if i == 0:
                df_all = results[i].get().copy()
            else:
                df_all = pd.concat([df_all, results[i].get()], axis=0)
                df_all.reset_index()

        corpus_embeddings = df_all['embeddings'].tolist()
        corpus = df_all['text'].tolist()

        generate_pickle(df_all_file, df_all)
        generate_pickle(corpus_embeddings_file, corpus_embeddings)
        generate_pickle(corpus_file, corpus)


    corpus_embeddings = np.array(corpus_embeddings)
    corpus_embeddings = np.squeeze(corpus_embeddings, axis=1)

    print("get embedding finish!")
    print("corpus embeddings shape:", corpus_embeddings.shape)

    return df_all, corpus_embeddings, corpus

def text_cluster(corpus_embeddings, corpus):
    print('Start Kmeans...')
    num_clusters = 50
    # CLUSTER_FILE =

    # clustering_model = KMeans(n_clusters=num_clusters)
    # clustering_model.fit(corpus_embeddings)


    # print('save models...')
    # generate_pickle("./data/clustering_model50.pkl", clustering_model)


    print('verify model...')
    clustering_model = load_pickle("./data/clustering_model50.pkl")
    cluster_assignment = clustering_model.labels_
    print(clustering_model.predict(corpus_embeddings[:10]))
    # # 每个样本所属的簇
    # label = []  # 存储1000个类标 4个类
    # # print(clf.labels_)
    # i = 0
    # print("save cluster result...")
    # clustered_sentences = [[] for i in range(num_clusters)]
    # # with open(f'./output/result_{time.strftime("%Y%m%d%H%I%S")}.txt', 'w') as fw:
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     clustered_sentences[cluster_id].append(corpus[sentence_id])
    #
    # generate_pickle("./output/cluster_result.pkl",  clustered_sentences)

    clustered_sentences = load_pickle("./output/cluster_result.pkl")
    # f = open("./output/cluster_result50.txt", 'wt')
    for i, cluster in enumerate(clustered_sentences):
        # print("Cluster ", i)
        # f.write("Cluster " + str(i) + '\n')

        if i != 0:
            continue

        for j in range(len(cluster)):
            print(cluster[j])
        if i != 0:
            break
            # f.write(cluster[j] + '\n')
    # f.close()

    '''
    cluster_centers_: [[ 0.93605761  0.59323462  0.99695483 ... -0.94325057 -0.93355352
       0.14013512]
     [ 0.88884779  0.80916959  0.99517786 ... -0.98770406 -0.90627313
      -0.35022781]
     [ 0.81379442  0.73286633  0.99625387 ... -0.95724261 -0.68449021
       0.01498421]
     ...
     [ 0.90434844 -0.57967657  0.99340826 ... -0.85602668 -0.69682665
      -0.38037719]
     [ 0.94694114  0.05898336  0.99690505 ... -0.92151306 -0.89468088
      -0.15717479]
     [ 0.84025225  0.82946124  0.99601236 ... -0.98372858 -0.80762365
      -0.42076355]]
    inertia: 28037084.180233765
    '''


    # print("cluster_centers_:", clustering_model.cluster_centers_)
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数  958.137281791
    # print("inertia:", clustering_model.inertia_)
    # 轮廓系数用两个指标来同时评估样本的簇内差异和簇间差异,越大越好
    # print("silhouette_score:",silhouette_score(test_tfidf, clf.labels_))
    pass

def cluster_res_display(corpus_embeddings, corpus):
    clustering_model = load_pickle("./data/clustering_model50.pkl")
    cluster_assignment = clustering_model.labels_
    print(clustering_model.predict(corpus_embeddings[:10]))
    # 每个样本所属的簇
    label = []  # 存储1000个类标 4个类
    num_clusters = 50
    # print(clf.labels_)
    i = 0
    print("save cluster result...")
    clustered_sentences = [[] for i in range(num_clusters)]
    # with open(f'./output/result_{time.strftime("%Y%m%d%H%I%S")}.txt', 'w') as fw:
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    generate_pickle("./output/cluster_result.pkl",  clustered_sentences)
    clustered_sentences = load_pickle("./output/cluster_result.pkl")

    f = open("./cluster_result.txt", 'wt')
    for i, cluster in enumerate(clustered_sentences):
        # print("Cluster ", i)
        f.write("Cluster " + str(i) + '\n')
        for j in range(len(cluster)):
            # print(cluster[j])
            f.write(cluster[j] + '\n')

    f.close()

def infer_cluster(corpus_embeddings):
    # clf = pickle.load(open('./models/twitter_spider/cluster50_twitter_spider.pkl', "rb"))
    # 942965
    clustering_model = load_pickle("./data/clustering_model50.pkl")
    clustering_labels = clustering_model.predict(corpus_embeddings)

    this_content_time_name = "tt_content_dict_pp" + str(increment_time) + ".pkl"
    # if off_mod == True:
    tt_content_dict = load_pickle("./output/" + this_content_time_name)

    print("聚类结果尺寸：", clustering_labels.shape)
    print("内容字典尺寸：", len(tt_content_dict))

    for i, (ck, cv) in enumerate(tqdm(tt_content_dict.items())):
        tt_content_dict[ck][-1] = clustering_labels[i]

    generate_pickle("./output/tt_content_cluster_dict.pkl",  tt_content_dict)

    pass

def analysis_text_cluster():
    clustered_sentences = load_pickle("./output/cluster_result.pkl")
    # f = open("./output/cluster_result50.txt", 'wt')

    check_cluster_num = 33
    check_count = 0
    for i, cluster in enumerate(clustered_sentences):
        # print("Cluster ", i)
        # f.write("Cluster " + str(i) + '\n')

        if i != check_cluster_num:
            continue

        for j in range(len(cluster)):
            this_line = cluster[j]
            if "习" in this_line:
                check_count += 1
                print(j, this_line)
        if i != check_cluster_num:
            break

    print("Cluster", check_cluster_num)
    print("类别关键词：")
    print("yh分布情况： 没有")
    print("输出检查个数：", check_count)
    print("总检查个数：", len(clustered_sentences[check_cluster_num]))
    print("假定yh比例：", float(check_count/len(clustered_sentences[check_cluster_num])))


if __name__ == '__main__':

    increment_time = 0

    data_list = get_normal_corpus(off_mod=True)
    # df_all, corpus_embeddings, corpus = get_text_embedding(data_list=None, off_mod=True)
    df_all, corpus_embeddings, corpus = get_text_embedding(data_list, off_mod=True)
    # text_cluster(corpus_embeddings, corpus)

    cluster_res_display(corpus_embeddings, corpus)

    # analysis_text_cluster()

    # infer_cluster(corpus_embeddings)



