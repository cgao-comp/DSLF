import numpy as np
import setproctitle
import torch
import create_graphs
import random
import networkx as nx
from dataReader import DataReader_snapshot
from dataLodaer import GraphData
from torch.utils.data import DataLoader
from model import GCN
from model import GAT
from GCN_EN import self_loop_attention_GCN
from GCN_EN import VAE
import torch.nn.functional as F
from event_ana import DataReader
from multiprocessing.dummy import Pool as Pool
import heapq
import pickle
import time
from tqdm import tqdm
import math
global F_score_global
global exe_time
global F_score_global_test
global exe_time_test
import args
import torch.nn as nn
with open('two_channel_twitter15_16.pkl', 'rb') as f:
    c1_twitter15_16 = pickle.load(f)
with open('two_channel_weibo.pkl', 'rb') as f:
    c2_weibo = pickle.load(f)
Twitter_propagation_cascade = []
for i, one_DAG in enumerate(tqdm(c1_twitter15_16.data['propagation_DAG'])):
    Twitter_propagation_cascade.append(one_DAG)
Weibo_propagation_cascade = []
for i, one_DAG in enumerate(tqdm(c2_weibo.data['propagation_DAG'])):
    Weibo_propagation_cascade.append(one_DAG)
del Twitter_propagation_cascade
del Weibo_propagation_cascade  
del c1_twitter15_16
del c2_weibo
with open('networkx_Twitter_union_graph_inf.pkl', 'rb') as f:
    Twitter_union_graph_inf = pickle.load(f)
with open('networkx_Weibo_union_graph_inf.pkl', 'rb') as f:
    Weibo_union_graph_inf = pickle.load(f)
ten_percent = int(0.9 * len(Twitter_propagation_cascade))
subset_Twitter = Twitter_propagation_cascade[:ten_percent]
subset_Weibo = Weibo_propagation_cascade[:ten_percent]
all_extend_subG_Twitter = []
for index, training_cascade in enumerate(tqdm(subset_Twitter)):
    try:
        one_hop_neighbors = set()
        for node in training_cascade.nodes():
            one_hop_neighbors.update(Twitter_union_graph_inf.neighbors(node))
        all_nodes = set(training_cascade.nodes()) | one_hop_neighbors
        subgraph = Twitter_union_graph_inf.subgraph(all_nodes).copy()
    except nx.NetworkXError as e:
        print(f"Error: {e}. Skipping this graph.")
        continue
    for node in training_cascade.nodes():
        subgraph.nodes[node]['time'] = training_cascade.nodes[node]['time']
    for node in one_hop_neighbors:
        if 'time' not in subgraph.nodes[node]:
            subgraph.nodes[node]['time'] = 999999
    all_extend_subG_Twitter.append(subgraph)
all_extend_subG_Weibo = []
for index, training_cascade in enumerate(tqdm(subset_Weibo)):
    try:
        min_time = min(data['time'] for _, data in training_cascade.nodes(data=True))
        for node, data in training_cascade.nodes(data=True):
            normalized_time = (data['time'] - min_time) / 60  
            training_cascade.nodes[node]['time'] = round(normalized_time, 2)
        one_hop_neighbors = set()
        for node in training_cascade.nodes():
            one_hop_neighbors.update(Weibo_union_graph_inf.neighbors(node))
        all_nodes = set(training_cascade.nodes()) | one_hop_neighbors
        subgraph = Weibo_union_graph_inf.subgraph(all_nodes).copy()
        if len(subgraph.nodes) > 30000:
            all_extend_subG_Weibo.append(None)
            continue
    except nx.NetworkXError as e:
        print(f"Error: {e}. Skipping this graph.")
        all_extend_subG_Weibo.append(None)
        continue
    for node in training_cascade.nodes():
        subgraph.nodes[node]['time'] = training_cascade.nodes[node]['time']
    for node in one_hop_neighbors:
        if 'time' not in subgraph.nodes[node]:
            subgraph.nodes[node]['time'] = 999999
    all_extend_subG_Weibo.append(subgraph)
all_extend_subG_Twitter_test = []
all_extend_subG_Weibo_test = []
for idx in range(len(all_extend_subG_Twitter)):
    if len(all_extend_subG_Twitter[idx].nodes)>3000 or all_extend_subG_Weibo[idx] is None or len(all_extend_subG_Weibo[idx].nodes)>3000:
        continue
    else:
        all_extend_subG_Twitter_test.append(all_extend_subG_Twitter[idx])
        all_extend_subG_Weibo_test.append(all_extend_subG_Weibo[idx])
del all_extend_subG_Twitter
del all_extend_subG_Weibo
rnd_state = np.random.RandomState(1111)
datareader_Twitter = DataReader_snapshot(all_extend_subG_Twitter_test,
                                rnd_state=rnd_state,
                                folds=10,
                                union_graph_inf = Twitter_union_graph_inf)
del all_extend_subG_Twitter_test
del Twitter_union_graph_inf
datareader_Weibo = DataReader_snapshot(all_extend_subG_Weibo_test,
                                rnd_state=rnd_state,
                                folds=10,
                                union_graph_inf = Weibo_union_graph_inf)
del all_extend_subG_Weibo_test
del Weibo_union_graph_inf
def collate_batch(batch):
    B = len(batch)  
    Chanels = batch[0][-2].shape[1]  
    N_nodes_max = batch[0][-2].shape[0]
    A = torch.zeros(1, N_nodes_max, N_nodes_max)
    A[0, :, :] = batch[0][0]
    x = torch.zeros(B, 5, N_nodes_max, Chanels)
    P = torch.zeros(B, 5, N_nodes_max)
    labels = torch.zeros(B, batch[0][1].shape[0])
    for b in range(B):
        x[b, 0, :] = batch[b][3]
        x[b, 1, :] = batch[b][4]
        x[b, 2, :] = batch[b][5]
        x[b, 3, :] = batch[b][6]
        x[b, 4, :] = batch[b][7]
        P[b, :, :] = batch[b][2]
        labels[b, :] = batch[b][1]
    N_nodes = torch.from_numpy(np.array(N_nodes_max)).long()
    influence = batch[0][8]
    return [A, labels, P, x, N_nodes, influence]
def F_score_computation(pred, labels_mul_hot, source_num, verNum):  
    assert pred.shape[2] == labels_mul_hot.shape[1], 'error'
    assert pred.shape[1] == 2, 'error'
    assert pred.shape[0] == labels_mul_hot.shape[0], 'error'
    assert pred.shape[2] == verNum, 'error'
    assert len(pred.shape) == 3, 'error'
    assert len(labels_mul_hot.shape) == 2, 'error'
    B = pred.shape[0]
    F_score_total = 0
    pred = torch.transpose(pred, 1, 2)
    for i in range(B):
        F_score_one_B = 0
        pred_one_B = pred[i, :, 1]  
        labels_mul_hot_one_B = labels_mul_hot[i, :] 
        tmp = zip(range(len(pred_one_B.tolist())), pred_one_B.tolist())
        largeN = heapq.nlargest(source_num, tmp, key=lambda x: x[1])
        for YZ in largeN:
            if labels_mul_hot_one_B[YZ[0]] == 1:
                F_score_one_B = F_score_one_B + 1 / (source_num)
        F_score_total += F_score_one_B
    return F_score_total/B
def train(train_loader_Twitter, train_loader_Weibo):
    global F_score_global, exe_time
    args.device = 'cuda'
    start = time.time()
    train_loss, n_samples, F_score_total = 0, 0, 0
    batch_idx = -1
    for data_twitter, data_weibo in zip(train_loader_Twitter, train_loader_Weibo):
        batch_idx = batch_idx + 1
        opt.zero_grad()
        B = data_twitter[3].shape[0]
        network_verNum_Twitter = data_twitter[4]
        network_verNum_Weibo = data_weibo[4]
        for i in range(len(data_twitter)):
            data_twitter[i] = data_twitter[i].to(args.device)
            data_weibo[i] = data_weibo[i].to(args.device)
        labels = data_twitter[1]  
        labels_mul_hot = labels.to(args.device)
        labels2 = data_weibo[1]  
        labels_mul_hot2 = labels2.to(args.device)
        out_ = torch.zeros(data_twitter[3].shape[0], data_twitter[3].shape[1], data_twitter[3].shape[2],
                           data_twitter[3].shape[3]).to(args.device)
        out2_ = torch.zeros(data_weibo[3].shape[0], data_weibo[3].shape[1], data_weibo[3].shape[2],
                           data_weibo[3].shape[3]).to(args.device)
        sorted_indices_twitter = torch.argsort(data_twitter[5], descending=True)
        sorted_indices_weibo = torch.argsort(data_weibo[5], descending=True)
        align_number = min(len(sorted_indices_twitter), len(sorted_indices_weibo))
        pred = torch.zeros(B, 5, align_number, 4).to(args.device)
        pred2 = torch.zeros(B, 5, align_number, 4).to(args.device)
        for bi in range(B):
            for ti in range(5):
                out_[bi, ti, :, :] = Attention_GCN1(data_twitter[0][bi, :, ], data_twitter[3][bi, ti, :, :])
                out2_[bi, ti, :, :] = Attention_GCN2(data_weibo[0][bi, :, ], data_weibo[3][bi, ti, :, :])
        out_[:, :, :, 12:] = (out_[:, :, :, 12:] + data_twitter[3][:, :, :, 12:]) / 2
        out2_[:, :, :, 12:] = (out2_[:, :, :, 12:] + data_weibo[3][:, :, :, 12:]) / 2
        for i in range(align_number):   
            v_index = sorted_indices_twitter[i]
            v_index2 = sorted_indices_weibo[i]
            pred[:, :, i, :], _ = LSTM_model1(out_[:, :, v_index, :])
            pred2[:, :, i, :], _ = LSTM_model2(out2_[:, :, v_index2, :])
        def compute_kl_loss(p, q, pad_mask=None):
            p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
            q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
            if pad_mask is not None:
                p_loss.masked_fill_(pad_mask, 0.)
                q_loss.masked_fill_(pad_mask, 0.)
            p_loss = p_loss.mean()
            q_loss = q_loss.mean()
            loss = (p_loss + q_loss) / 2
            return loss
        pred_reshaped1 = pred.permute(0, 2, 1, 3)  
        pred_reshaped2 = pred2.permute(0, 2, 1, 3)  
        def positional_encoding(time_steps, channels):
            position = torch.arange(time_steps).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
            pe = torch.zeros(time_steps, channels)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe
        T_, C_ = pred_reshaped1.shape[2], pred_reshaped1.shape[3]
        pe = positional_encoding(T_, C_).to('cuda')
        pred_with_pe1 = pred_reshaped1 + pe
        pred_with_pe2 = pred_reshaped2 + pe
        pred_with_pe1 = pred_with_pe1.squeeze(0)
        pred_with_pe2 = pred_with_pe2.squeeze(0)
        z1, decoder_z1, mu1, logvar1 = vae1(pred_with_pe1.reshape(align_number, 5 * 4))
        z2, decoder_z2, mu2, logvar2 = vae2(pred_with_pe2.reshape(align_number, 5 * 4))
        kl_loss_cross = compute_kl_loss(z1.squeeze(0), z2.squeeze(0))
        input_data_reshaped1 = pred_with_pe1.reshape(align_number, 5 * 4)
        input_data_reshaped2 = pred_with_pe2.reshape(align_number, 5 * 4)
        vae_loss1 = vae1.vae_loss_function(decoder_z1, input_data_reshaped1, mu1, logvar1)
        vae_loss2 = vae2.vae_loss_function(decoder_z2, input_data_reshaped2, mu2, logvar2)
        vae_loss = (vae_loss1 + vae_loss2) / 2 /20
        forward_out = pred[:, :, :, :2]
        backward_out = pred[:, :, :, 2:]
        average_out = (forward_out + backward_out) / 2.0   
        average_over_sequence = torch.mean(average_out, dim=1)  
        forward_out2 = pred2[:, :, :, :2]
        backward_out2 = pred2[:, :, :, 2:]
        average_out2 = (forward_out2 + backward_out2) / 2.0   
        average_over_sequence2 = torch.mean(average_out2, dim=1)  
        def weight_loss(pred_re, sourceNum, label_hot):
            B = pred_re.shape[0]
            loss_total = torch.tensor([0.], ).to(args.device)
            weight_I = 0.9
            weight_S = 0.1
            for i in range(B):
                pred_S_one_B = pred_re[i, 0, :]  
                pred_I_one_B = pred_re[i, 1, :]  
                loss_total = loss_total - (weight_I * sum(label_hot[i] * torch.log(pred_I_one_B)) +
                                           weight_S * sum((1 - label_hot[i]) * torch.log(pred_S_one_B))) / (pred_re.shape[2])
            loss_total = loss_total / B
            return loss_total
        labels_mul_hot_twitter_aligned = torch.index_select(labels_mul_hot, 1,
                                                            sorted_indices_twitter[:align_number])
        labels_mul_hot_weibo_aligned = torch.index_select(labels_mul_hot2, 1, sorted_indices_weibo[:align_number])
        average_over_sequence = torch.transpose(average_over_sequence, 1, 2)  
        weights = torch.tensor([1/data_twitter[4], 1-1/data_twitter[4]]).to(args.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        pred_re_reshaped = average_over_sequence.transpose(1, 2).reshape(-1, 2)
        labels_reshaped = labels_mul_hot_twitter_aligned.view(-1).long()
        loss = criterion(pred_re_reshaped, labels_reshaped)
        average_over_sequence2 = torch.transpose(average_over_sequence2, 1, 2)  
        weights2 = torch.tensor([1/data_weibo[4], 1-1/data_weibo[4]]).to(args.device)
        criterion2 = nn.CrossEntropyLoss(weight=weights2)
        pred_re_reshaped2 = average_over_sequence2.transpose(1, 2).reshape(-1, 2)
        labels_reshaped2 = labels_mul_hot_weibo_aligned.view(-1).long()
        loss2 = criterion2(pred_re_reshaped2, labels_reshaped2)
        loss_total = (loss+loss2) + kl_loss_cross + vae_loss
        loss_total.backward()
        opt.step()
        time_iter = time.time() - start
        train_loss += loss_total.item() * len(out_)
        n_samples += len(out_)
        F_score_global += len(out_) * max(F_score_computation(average_over_sequence, labels_mul_hot_twitter_aligned,
                                    source_num=1, verNum=align_number), 
                                       F_score_computation(average_over_sequence2, labels_mul_hot_weibo_aligned,
                                                           source_num=1, verNum=align_number) )
        exe_time += 1
        if batch_idx % 10 == 0 or batch_idx == len(train_loader_Twitter) - 1 or 0 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f})\tF-score:{:.4f} \tsecond/iteration: {:.4f}'.format(
                epoch + 1, n_samples, len(train_loader_Twitter.dataset),
                100. * (batch_idx + 1) / len(train_loader_Twitter), loss.item(), train_loss / n_samples, F_score_global / exe_time,
                time_iter / (batch_idx + 1)))
def Test(train_loader_Twitter, train_loader_Weibo):
    global F_score_global_test, exe_time_test
    args.device = 'cuda'
    start = time.time()
    train_loss, n_samples, F_score_total = 0, 0, 0
    batch_idx = -1
    for data_twitter, data_weibo in zip(train_loader_Twitter, train_loader_Weibo):
        batch_idx = batch_idx + 1
        B = data_twitter[3].shape[0]
        network_verNum_Twitter = data_twitter[4]
        network_verNum_Weibo = data_weibo[4]
        for i in range(len(data_twitter)):
            data_twitter[i] = data_twitter[i].to(args.device)
            data_weibo[i] = data_weibo[i].to(args.device)
        labels = data_twitter[1]  
        labels_mul_hot = labels.to(args.device)
        labels2 = data_weibo[1]  
        labels_mul_hot2 = labels2.to(args.device)
        out_ = torch.zeros(data_twitter[3].shape[0], data_twitter[3].shape[1], data_twitter[3].shape[2],
                           data_twitter[3].shape[3]).to(args.device)
        out2_ = torch.zeros(data_weibo[3].shape[0], data_weibo[3].shape[1], data_weibo[3].shape[2],
                           data_weibo[3].shape[3]).to(args.device)
        sorted_indices_twitter = torch.argsort(data_twitter[5], descending=True)
        sorted_indices_weibo = torch.argsort(data_weibo[5], descending=True)
        align_number = min(len(sorted_indices_twitter), len(sorted_indices_weibo))
        pred = torch.zeros(B, 5, align_number, 4).to(args.device)
        pred2 = torch.zeros(B, 5, align_number, 4).to(args.device)
        for bi in range(B):
            for ti in range(5):
                out_[bi, ti, :, :] = Attention_GCN1(data_twitter[0][bi, :, ], data_twitter[3][bi, ti, :, :])
                out2_[bi, ti, :, :] = Attention_GCN2(data_weibo[0][bi, :, ], data_weibo[3][bi, ti, :, :])
        out_[:, :, :, 12:] = (out_[:, :, :, 12:] + data_twitter[3][:, :, :, 12:]) / 2
        out2_[:, :, :, 12:] = (out2_[:, :, :, 12:] + data_weibo[3][:, :, :, 12:]) / 2
        for i in range(align_number):   
            v_index = sorted_indices_twitter[i]
            v_index2 = sorted_indices_weibo[i]
            pred[:, :, i, :], _ = LSTM_model1(out_[:, :, v_index, :])
            pred2[:, :, i, :], _ = LSTM_model2(out2_[:, :, v_index2, :])
        forward_out = pred[:, :, :, :2]
        backward_out = pred[:, :, :, 2:]
        average_out = (forward_out + backward_out) / 2.0   
        average_over_sequence = F.softmax(torch.mean(average_out, dim=1), dim=2)  
        forward_out2 = pred2[:, :, :, :2]
        backward_out2 = pred2[:, :, :, 2:]
        average_out2 = (forward_out2 + backward_out2) / 2.0   
        average_over_sequence2 = F.softmax(torch.mean(average_out2, dim=1), dim=2)  
        labels_mul_hot_twitter_aligned = torch.index_select(labels_mul_hot, 1,
                                                            sorted_indices_twitter[:align_number])
        labels_mul_hot_weibo_aligned = torch.index_select(labels_mul_hot2, 1, sorted_indices_weibo[:align_number])
        time_iter = time.time() - start
        n_samples += len(out_)
        average_over_sequence2 = torch.transpose(average_over_sequence2, 1, 2)
        average_over_sequence = torch.transpose(average_over_sequence, 1, 2)
        F_score_global_test += len(out_) * max(F_score_computation(average_over_sequence, labels_mul_hot_twitter_aligned,
                                    source_num=1, verNum=align_number), 
                                       F_score_computation(average_over_sequence2, labels_mul_hot_weibo_aligned,
                                                           source_num=1, verNum=align_number) )
        exe_time_test += 1
        if batch_idx % 10 == 0 or batch_idx == len(train_loader_Twitter) - 1 or 0 == 0:
            print('*Test Epoch: {} [{}/{} ({:.0f}%)]\tF-score:{:.4f} \tsecond/iteration: {:.4f}'.format(
                epoch + 1,
                n_samples,
                len(train_loader_Twitter.dataset),
                100. * (batch_idx + 1) / len(train_loader_Twitter),
                F_score_global_test / exe_time_test,
                time_iter / (batch_idx + 1)))
n_folds = 1
for fold_id in range(n_folds):
    loaders_Twitter = []
    loaders_Weibo = []
    for split in ['train', 'test']:
        gdata_Twitter = GraphData(fold_id=fold_id,
                          datareader=datareader_Twitter,  
                          split=split)
        loader_Twitter = DataLoader(gdata_Twitter,  
                            batch_size=1,  
                            shuffle=False,  
                            num_workers=4,
                            collate_fn=collate_batch)  
        loaders_Twitter.append(loader_Twitter)  
        gdata_Weibo = GraphData(fold_id=fold_id,
                          datareader=datareader_Weibo,  
                          split=split)
        loader_Weibo = DataLoader(gdata_Weibo,  
                            batch_size=1,  
                            shuffle=False,  
                            num_workers=4,
                            collate_fn=collate_batch)  
        loaders_Weibo.append(loader_Weibo)
    print('\nTwitter: FOLD {}/{}, train {}, test {}'.format(fold_id + 1, n_folds, len(loaders_Twitter[0].dataset),
                                                   len(loaders_Twitter[1].dataset)))
    print('\nWeibo: FOLD {}/{}, train {}, test {}'.format(fold_id + 1, n_folds, len(loaders_Weibo[0].dataset),
                                                   len(loaders_Weibo[1].dataset)))
    LSTM_model1 = nn.LSTM(15, hidden_size=2, num_layers=3, batch_first=True,
                        bidirectional=True).to('cuda')
    Attention_GCN1 = self_loop_attention_GCN(15, 4).to('cuda')
    LSTM_model2 = nn.LSTM(15, hidden_size=2, num_layers=3, batch_first=True,
                        bidirectional=True).to('cuda')
    Attention_GCN2 = self_loop_attention_GCN(15, 4).to('cuda')
    num_params1 = sum(p.numel() for p in Attention_GCN1.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params1}")
    num_params2 = sum(p.numel() for p in LSTM_model2.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params2}")
    vae1 = VAE(20, 16).to('cuda')
    vae2 = VAE(20, 16).to('cuda')
    num_params3 = sum(p.numel() for p in vae1.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params3}")
    opt = torch.optim.Adam([
        {'params': LSTM_model1.parameters(), 'lr': 0.0005},
        {'params': Attention_GCN1.parameters(), 'lr': 0.0005},
        {'params': LSTM_model2.parameters(), 'lr': 0.0005},
        {'params': Attention_GCN2.parameters(), 'lr': 0.0005},
        {'params': vae1.parameters(), 'lr': 0.0005},
        {'params': vae2.parameters(), 'lr': 0.0005},
    ])
    for epoch in range(args.epochs):
        LSTM_model1.train()
        Attention_GCN1.train()
        LSTM_model2.train()
        Attention_GCN2.train()
        vae1.train()
        vae2.train()
        F_score_global = 0.0
        exe_time = 0.0
        F_score_global_test = 0.0
        exe_time_test = 0.0
        train(loaders_Twitter[0], loaders_Weibo[0])
        Test(loaders_Twitter[1], loaders_Weibo[1])

