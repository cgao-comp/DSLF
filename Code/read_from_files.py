def read_and_sort_similarity_file(file_path):
  
    index = 0
    all_similarities = []
    with open(file_path, 'r') as file:
        for line in file:
            index = index + 1
            print(index)
            twitter_eid, weibo_eid, similarity = line.strip().split()
            similarity = float(similarity)
            all_similarities.append((twitter_eid, weibo_eid, similarity))
    print('排序中')
    all_similarities.sort(key=lambda x: x[2], reverse=True)
    print('排序完成')
    return all_similarities
def find_best_unique_matches(all_similarities):
    
    matched_twitter = set()
    matched_weibo = set()
    best_matches = []
    for twitter_eid, weibo_eid, similarity in all_similarities:
        if twitter_eid not in matched_twitter and weibo_eid not in matched_weibo:
            best_matches.append((twitter_eid, weibo_eid, similarity))
            matched_twitter.add(twitter_eid)
            matched_weibo.add(weibo_eid)
    return best_matches
import pickle
from event_ana import DataReader
with open('weibo_v4comments.pkl', 'rb') as f:
    weibo = pickle.load(f)
with open('twitter15_comments.pkl', 'rb') as f:
    twitter15 = pickle.load(f)
with open('twitter16_comments.pkl', 'rb') as f:
    twitter16 = pickle.load(f)
print("文件读取完成.....")
file_path = 'pipei.txt'
twitter_to_weibo = {}
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        twitter_eid = parts[0].strip("(' ")
        weibo_eid = parts[1].strip(" '")
        twitter_to_weibo[twitter_eid] = weibo_eid
print(twitter_to_weibo)
print('两平台事件配对完成.....')
import networkx as nx
twitter15_16 = twitter15
uniongraph = nx.compose(twitter15_16.data['unionGraph'], twitter16.data['unionGraph'])
largest_nodes_set = max(nx.connected_components(uniongraph), key=len)
largest_component = uniongraph.subgraph(largest_nodes_set)
twitter15_16.data['largest_component'] = largest_component
for i in twitter16.data['event_id']:
    twitter15_16.data['event_id'].append(i)
for i in twitter16.data['source']:
    twitter15_16.data['source'].append(i)
unionGraph = []
unionGraph.append(twitter15_16.data['unionGraph'])
unionGraph.append(twitter16.data['unionGraph'])
twitter15_16.data['unionGraph']=unionGraph
for i in twitter16.data['propagation']:
    twitter15_16.data['propagation'].append(i)
for i in twitter16.data['Fake_or_True']:
    twitter15_16.data['Fake_or_True'].append(i)
for k,v in twitter16.data['user'].items():
    twitter15_16.data['user'][k]=v
for i in twitter16.data['propagation_DAG']:
    twitter15_16.data['propagation_DAG'].append(i)
for i in twitter16.data['comment']:
    twitter15_16.data['comment'].append(i)
print('15和16合并完成....')
import copy
from tqdm import tqdm
copied_twitter15_16 = copy.deepcopy(twitter15_16)
copied_weibo = copy.deepcopy(weibo)
print('深copy完成.......')
copied_twitter15_16.data['event_id'] = []
copied_twitter15_16.data['source'] = []
copied_twitter15_16.data['propagation'] = []
copied_twitter15_16.data['Fake_or_True'] = []
copied_twitter15_16.data['propagation_DAG'] = []
copied_twitter15_16.data['comment'] = []
copied_weibo.data['event_id'] = []
copied_weibo.data['source'] = []
copied_weibo.data['propagation'] = []
copied_weibo.data['Fake_or_True'] = []
copied_weibo.data['propagation_DAG'] = []
copied_weibo.data['comment'] = []
copied_weibo.data['comments'] = []
index_all = 0
for twitter_, weibo_ in tqdm(twitter_to_weibo.items()):
    index_all += 1
    index_twitter = 0
    index_weibo = 0
    while twitter15_16.data['event_id'][index_twitter] != twitter_:
        index_twitter = index_twitter + 1
    while weibo.data['event_id'][index_weibo] != weibo_:
        index_weibo = index_weibo + 1
    copied_twitter15_16.data['event_id'].append(twitter15_16.data['event_id'][index_twitter])
    copied_twitter15_16.data['source'].append(twitter15_16.data['source'][index_twitter])
    copied_twitter15_16.data['propagation'].append(twitter15_16.data['propagation'][index_twitter])
    copied_twitter15_16.data['Fake_or_True'].append(twitter15_16.data['Fake_or_True'][index_twitter])
    copied_twitter15_16.data['propagation_DAG'].append(twitter15_16.data['propagation_DAG'][index_twitter])
    copied_twitter15_16.data['comment'].append(twitter15_16.data['comment'][index_twitter])
    copied_weibo.data['event_id'].append(weibo.data['event_id'][index_weibo])
    copied_weibo.data['source'].append(weibo.data['source'][index_weibo])
    copied_weibo.data['propagation'].append(weibo.data['propagation'][index_weibo])
    copied_weibo.data['Fake_or_True'].append(weibo.data['Fake_or_True'][index_weibo])
    copied_weibo.data['propagation_DAG'].append(weibo.data['propagation'][index_weibo])
    copied_weibo.data['propagation'].append(weibo.data['propagation'][index_weibo].to_undirected())
    copied_weibo.data['comment'].append(weibo.data['comments'][index_weibo])
with open('two_channel_twitter15_16.pkl', 'wb') as f:
    pickle.dump(copied_twitter15_16, f)
with open('two_channel_weibo.pkl', 'wb') as f:
    pickle.dump(copied_weibo, f)
print('两平台有效事件数据集处理完成，已对齐.....')
print('------------------------------------------------')