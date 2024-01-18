import numpy as np
import sklearn
from os.path import join as pjoin
import os
import networkx as nx
import sys
import pickle
import json
import re
class DataReader():
    def __init__(self,
                 data_dir,  
                 rnd_state=None):
        self.data_dir = data_dir
        files = os.listdir(self.data_dir)
        self.data = {}  
        self.data['reindex_from0'] = []   
        self.data['event_id']=[]
        self.data['source'] = []
        self.tree_index = 1
        self.data['unionGraph'] = nx.Graph()
        self.data['largest_component'] = nx.Graph()
        self.data['propagation'] = []
        self.data['Fake_or_True'] = []  
        self.data['user']={}
        self.data['diameter']=[]
        self.data['comments'] = []
        file_count = 1
        for file_name in files:
            print(file_count)
            file_count+=1
            if file_name in ['3495745049431351.json']:
                continue
            self.data['event_id'].append(file_name.replace(".json", ""))
            propag_tree = self.propagation_tree_processing(list(filter(lambda f: f.find(file_name) >= 0, files))[0])
            assert len(propag_tree.nodes)-len(propag_tree.edges)==1, 'cuo'
            self.data['propagation'].append(propag_tree)
            self.unionGraphCons(list(filter(lambda f: f.find(file_name) >= 0, files))[0])
            comments = self.getComment(list(filter(lambda f: f.find(file_name) >= 0, files))[0])
            self.data['comments'].append(comments)
        print()
        self.getlabel('Weibo.txt')
        print()
    def getComment(self, fpath):
        all_comments = []
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = json.load(f)
        for each_user in lines:
            the_text = each_user['text']
            if the_text != '' and the_text != '轉發微博' and the_text != '转发微博':
                all_comments.append(the_text)
        return all_comments
    def getlabel(self, fpath):
        with open(fpath, encoding='utf-8') as f:
            lines = f.readlines()
        for id in self.data['event_id']:
            for line in lines:
                line_split = line.split('\t')
                line_eid = line_split[0].split(':')[1]
                line_label_10 = line_split[1].split(':')[1]
                if line_eid == id:
                    if line_label_10 == '1':
                        self.data['Fake_or_True'].append('rumor')
                    else:
                        self.data['Fake_or_True'].append('common')
                    break
    def unionGraphCons(self, fpath):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = json.load(f)
        uid_mid_dic = {}
        uid_parent_dic = {}
        for one_user in lines:
            uid_mid_dic[one_user['mid']] = [one_user['uid']]
            if one_user['uid'] in uid_parent_dic:
                uid_parent_dic[one_user['uid']].append(one_user['parent'])  
            else:
                uid_parent_dic[one_user['uid']] = [one_user['parent']]
        for one_user_pairs in uid_parent_dic.items():
            child_uid = int(one_user_pairs[0])
            his_parentMid_list = one_user_pairs[1]
            if None in his_parentMid_list:
                pass
            else:
                for his_each_parentMid_id in his_parentMid_list:
                    parent_uid = int(uid_mid_dic[his_each_parentMid_id][0])
                    if parent_uid == child_uid:
                        continue
                    self.data['unionGraph'].add_node(parent_uid)
                    self.data['unionGraph'].add_node(child_uid)
                    if 'att' not in self.data['unionGraph'].nodes[parent_uid]:
                        self.data['unionGraph'].nodes[parent_uid]['att'] = self.data['user'].get(parent_uid)
                    if 'att' not in self.data['unionGraph'].nodes[child_uid]:
                        self.data['unionGraph'].nodes[child_uid]['att'] = self.data['user'].get(child_uid)
                    if self.data['unionGraph'].has_edge(parent_uid, child_uid):
                        w = self.data['unionGraph'].edges[parent_uid, child_uid]['weight']
                        self.data['unionGraph'].add_edge(parent_uid, child_uid, weight=w + 1)  
                    else:
                        self.data['unionGraph'].add_edge(parent_uid, child_uid, weight=1)
    def get_max_G(self):
        largest = max(nx.connected_components(self.data['unionGraph']), key=len)
        self.data['largest_component'] = self.data['unionGraph'].subgraph(largest)
    def propagation_tree_processing(self, fpath): 
        g = nx.DiGraph()
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = json.load(f)
        uid_mid_dic = {}
        uid_parent_dic = {}
        uid_EARtime_dic = {}
        for one_user in lines:
            uid_mid_dic[one_user['mid']] = [one_user['uid'], one_user['t']]
            if one_user['uid'] in uid_parent_dic:
                uid_parent_dic[one_user['uid']].append(one_user['parent'])  
            else:
                uid_parent_dic[one_user['uid']] = [one_user['parent']]
            if one_user['uid'] in uid_EARtime_dic:
                if uid_EARtime_dic[one_user['uid']] > one_user['t']:
                    uid_EARtime_dic[one_user['uid']] = one_user['t']
            else:
                uid_EARtime_dic[one_user['uid']] = one_user['t']
            if one_user['uid'] not in self.data['user']:
                self.data['user'][one_user['uid']] = [one_user['statuses_count'], one_user['user_created_at'], one_user['friends_count'], one_user['followers_count'], one_user['verified']]
        for one_user_pairs in uid_parent_dic.items():
            child_uid = one_user_pairs[0]
            his_parentMid_list = one_user_pairs[1]
            if None in his_parentMid_list:
                self.data['source'].append(child_uid)
            else:
                his_earliest_get_time = 999999999999
                form_whom_parent_uid = None
                for his_each_parentMid_id in his_parentMid_list:
                    parent_uid = uid_mid_dic[his_each_parentMid_id][0]
                    parent_m_t = uid_mid_dic[his_each_parentMid_id][1]
                    if parent_uid == child_uid:
                        continue
                    if parent_m_t < his_earliest_get_time:
                        his_earliest_get_time = parent_m_t
                        form_whom_parent_uid = parent_uid
                g.add_node(child_uid)
                g.add_node(form_whom_parent_uid)
                g.add_edge(form_whom_parent_uid,child_uid)
                if 'time' not in g.nodes[form_whom_parent_uid]:
                    g.nodes[form_whom_parent_uid]['time'] = his_earliest_get_time
                else:
                    if g.nodes[form_whom_parent_uid]['time'] > his_earliest_get_time:
                        g.nodes[form_whom_parent_uid]['time'] = his_earliest_get_time
                if 'time' not in g.nodes[child_uid]:
                    g.nodes[child_uid]['time'] = uid_EARtime_dic[child_uid]
        empty_nodes = [node for node in g.nodes() if g.nodes[node]['time'] is None]
        assert len(empty_nodes) ==0, '不是吧'
        return g
if __name__ == '__main__':
    with open('twitter15.pkl', 'rb') as f:
        twitter15 = pickle.load(f)
    twitter15.data['comment'] = []
    for i in range(len(twitter15.data['source'])):
        twitter15.data['comment'].append([])
    files = os.listdir('./datasets/t15/replies')
    dict_eid_comments = {}
    for file_one_name in files:
        with open(pjoin('./datasets/t15/replies', file_one_name), 'r') as f:
            tweets = f.readlines()
        def clean_tweet(tweet):
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'@\S+', '', tweet)
            tweet = re.sub(r'
            tweet = re.sub(r'\bRT\b', '', tweet)
            tweet = re.sub(r'[^\w\s]', '', tweet)
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            return tweet
        event_id = file_one_name.replace('.txt', '')
        cleaned_tweets_one_tree = [clean_tweet(tweet) for tweet in tweets if clean_tweet(tweet)]
        dict_eid_comments[event_id] = cleaned_tweets_one_tree
    print()
    for i, eid in enumerate(twitter15.data['event_id']):
        if eid in dict_eid_comments:
            twitter15.data['comment'][i] = dict_eid_comments[eid]
    with open('twitter15_comments.pkl', 'wb') as f:
        pickle.dump(twitter15, f)
    with open('twitter16.pkl', 'rb') as f:
        twitter16 = pickle.load(f)
    twitter16.data['comment'] = []
    for i in range(len(twitter16.data['source'])):
        twitter16.data['comment'].append([])
    files = os.listdir('./datasets/t16/replies')
    dict_eid_comments = {}
    for file_one_name in files:
        with open(pjoin('./datasets/t16/replies', file_one_name), 'r') as f:
            tweets = f.readlines()
        def clean_tweet(tweet):
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'@\S+', '', tweet)
            tweet = re.sub(r'
            tweet = re.sub(r'\bRT\b', '', tweet)
            tweet = re.sub(r'[^\w\s]', '', tweet)
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            return tweet
        event_id = file_one_name.replace('.txt', '')
        cleaned_tweets_one_tree = [clean_tweet(tweet) for tweet in tweets if clean_tweet(tweet)]
        dict_eid_comments[event_id] = cleaned_tweets_one_tree
    print()
    for i, eid in enumerate(twitter16.data['event_id']):
        if eid in dict_eid_comments:
            twitter16.data['comment'][i] = dict_eid_comments[eid]
    with open('twitter16_comments.pkl', 'wb') as f:
        pickle.dump(twitter16, f)
    print()
