import pickle
from event_ana import DataReader
print("开始加载数据")
with open('weibo_v4comments.pkl', 'rb') as f:
    weibo = pickle.load(f)
with open('twitter15_comments.pkl', 'rb') as f:
    twitter15 = pickle.load(f)
with open('twitter16_comments.pkl', 'rb') as f:
    twitter16 = pickle.load(f)
print("数据加载完毕")
from transformers import BertTokenizer, BertModel  
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
print('main....')
print('main....')
def get_text_embeddings(model, tokenizer, texts):
    embeddings = []
    for text in texts:
        encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            output = model(**encoded_input)
        cls_embedding = output.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding[0])
    return embeddings
local_model_path = './bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)
twitter_embedding_dict = {}
for index_twitter, comments_twitter in enumerate(tqdm(twitter15.data['comment'])):
    twitter_eid = twitter15.data['event_id'][index_twitter]
    if len(comments_twitter) == 0:
        continue
    twitter_comments_matrix = get_text_embeddings(model, tokenizer, comments_twitter)
    twitter_embedding_dict[twitter_eid] = twitter_comments_matrix
for index_twitter, comments_twitter in enumerate(tqdm(twitter16.data['comment'])):
    twitter_eid = twitter16.data['event_id'][index_twitter]
    if len(comments_twitter) == 0:
        continue
    twitter_comments_matrix = get_text_embeddings(model, tokenizer, comments_twitter)
    twitter_embedding_dict[twitter_eid] = twitter_comments_matrix
weibo_embedding_dict = {}
for index_weibo, comments_weibo in enumerate(tqdm(weibo.data['comments'])):
    weibo_eid = weibo.data['event_id'][index_weibo]
    if len(comments_weibo) == 0:
        continue
    weibo_comments_matrix = get_text_embeddings(model, tokenizer, comments_weibo)
    weibo_embedding_dict[weibo_eid] = weibo_comments_matrix
with open("output.txt", "w") as file:
    index = 0
    for t_eid, twitter_comments in twitter_embedding_dict.items():
        index = index + 1
        for w_eid, weibo_comments in weibo_embedding_dict.items():
            similarity_matrix = cosine_similarity(twitter_comments, weibo_comments)
            max_similarities1 = np.max(similarity_matrix, axis=1)
            max_similarities2 = np.max(similarity_matrix, axis=0)
            avg_max_similarity = (np.mean(max_similarities1) + np.mean(max_similarities2)) / 2
            content = f"{t_eid} {w_eid} {avg_max_similarity}\n"
            file.write(content)
            print(index, ': ', t_eid, ' ', w_eid, ' ', avg_max_similarity)
print("相似度计算完毕并且写入output.txt")