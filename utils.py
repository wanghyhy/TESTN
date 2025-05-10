import pickle
import numpy as np
import paddle
import pgl
import os
import random


def select_time_data(data, relation_dict, time_list):
    result = []
    for key in time_list:
        key='_'+key
        ids = [v for k, v in relation_dict.items() if key in k]
        data_selected = [data[data[:, 1]==i] for i in ids]
        result.append(paddle.concat(data_selected))
    return result

def adaptive_bin_distance(bin_num, coords, train):
    src_coord = coords[train[:, 0]] 
    dst_coord = coords[train[:, 2]] 
    delat = src_coord - dst_coord
    all_dist = np.sqrt((delat * delat).sum(1).numpy())

    all_dist = np.sort(all_dist)
    cnt = len(all_dist)
    res = []
    for i in range(0, cnt, int(cnt/bin_num)):
        res += [all_dist[i]]
    return paddle.to_tensor(res[1:])

def build_dynamic_labels(data, relation_dict, rel_list, time_list, num_nodes):
    pair_list, label_list = [[] for _ in range(len(time_list))], [[] for _ in range(len(time_list))]
    for rel in rel_list:
        g_list = [np.zeros((num_nodes, num_nodes)) for _ in range(len(time_list))]
        for i, time in enumerate(time_list):
            ids = [v for k, v in relation_dict.items() if rel in k and 'at_' + time in k]
            data_selected = [data[data[:, 1]==i] for i in ids]
            data_selected = np.concatenate(data_selected)
            g_list[i][data_selected[:,0], data_selected[:,2]] = 1
        for i in range(len(time_list)):
            src_t, dst_t = g_list[i].nonzero()
            j = (i + 1) % len(time_list)
            labels = g_list[j][src_t, dst_t]
            pair_list[i].append(np.stack([src_t, dst_t], 1))
            label_list[i].append(labels)

    pair_list = [paddle.to_tensor(np.concatenate(x)) for x in pair_list]
    label_list = [paddle.to_tensor(np.concatenate(x)).reshape([-1, 1]).cast('float32') for x in label_list]
    return pair_list, label_list

def generate_batch_grids(grid_dict, grid_batch_size, negative_ratio=2.0, sampling_hop=2):
    minx,maxx,miny,maxy=1000,-1000,1000,-1000
    for i in grid_dict.keys():
        minx=min(minx,i[0])
        maxx=max(maxx,i[0])
        miny=min(miny,i[1])
        maxy=max(maxy,i[1])
    all_grid = np.arange(len(grid_dict))
    grid_batch_size = min(len(all_grid), grid_batch_size)
    inds = np.random.choice(all_grid, grid_batch_size)
    keys = list(grid_dict.keys())
    batch_keys = [keys[i] for i in inds]
    pos_samples = np.concatenate([grid_dict[k] for k in batch_keys])
    neg_samples = negative_sampling_grid_near(batch_keys, grid_dict, negative_ratio, negative_ratio)
    grid_sizes = [len(grid_dict[k]) for k in batch_keys]
    labels = np.zeros(len(pos_samples) * (negative_ratio + 1), dtype=np.float32)
    labels[: len(pos_samples)] = 1
    return np.array(grid_sizes), np.array(pos_samples), np.array(neg_samples), np.array(labels)

def negative_sampling_grid_near(keys, grid_dict, sampling_hop, negative_ratio):
    all_nodes = set(np.concatenate([ll for ll in grid_dict.values()]))
    all_grids = list(grid_dict.keys())
    keys_expand = np.tile(np.expand_dims(keys, 1),(1,len(all_grids),1))
    manha = np.abs(keys_expand - all_grids).sum(-1)
    inds_x, inds_y = np.where((manha>sampling_hop) & (manha<sampling_hop+4))
    grids_expand = np.repeat(np.expand_dims(all_grids, 0), len(keys), axis=0)
    sampling_grids = grids_expand[inds_x, inds_y]

    neg_samples = []
    random_cnt = 0
    for i in range(len(keys)):
        neg_grids = sampling_grids[inds_x == i]
        if len(neg_grids) == 0:
            nodes_range = list(all_nodes - set(grid_dict[keys[i]]))
            random_cnt += 1
        else:
            nodes_range = np.concatenate([grid_dict[tuple(g)] for g in neg_grids])
        neg_nodes = np.random.choice(nodes_range, negative_ratio * len(grid_dict[keys[i]]))
        neg_samples.append(neg_nodes)
    return np.concatenate(neg_samples)

def build_hop2_dict(train_triplets, relation_dict, rel_list, time_list, dataset, path_len=2):
    cache_file = './runs/%s_path_len_%d.pickle' % (dataset, path_len)
    mid_file = './runs/%s_mid_file.pickle' % (dataset)
    time_relation_file = './runs/%s_time_relation.pickle' % (dataset)
    if os.path.exists(cache_file) and os.path.exists(mid_file) and os.path.exists(time_relation_file):
        with open(cache_file, 'rb') as f:
            hop2_dict = pickle.load(f)
        with open(mid_file, 'rb') as f:
            mid_dict = pickle.load(f)
        with open(time_relation_file, 'rb') as f:
            time_relation = pickle.load(f)
        return hop2_dict,mid_dict,time_relation

    print('Constructing two-hop dict...')
    hop2_dict = {}
    src, rel, dst = train_triplets.transpose()
    relation_dict_r = {v: int(k.split('_at_')[0]==rel_list[0]) for k, v in relation_dict.items()}
    time_relation={}
    mid_dict={}
    for i,time in enumerate(time_list):
        ids = [v for k, v in relation_dict.items() if 'at_' + time in k]
        ids = np.isin(rel, ids)
        src_t = src[ids]
        rel_t = rel[ids]
        dst_t = dst[ids]

        graph = {}
        graph4time = {}
        mid = {}
        dd = {'00': {}, '01': {}, '10': {}, '11': {}}
        for u, r, v in zip(src_t, rel_t, dst_t):
            if u not in graph:
                graph[u] = {}
                graph4time[u] = {}
            graph[u][v] = relation_dict_r[r]
            graph4time[u][v] = int(r/4)
            if u not in mid:
                mid[u] = {0:[],1:[]}
            mid[u][relation_dict_r[r]]+=[v]

        time_relation[i]=graph4time

        for u, u_N in graph.items():
            for v, r1 in u_N.items():
                for v2, r2 in graph[v].items():
                    if u != v2:
                        r12 = '00'
                        if (u,v2) not in dd[r12]:
                            dd[r12][(u,v2)] = []
                        dd[r12][(u,v2)] += [v]
                        

        mid_dict[time]=mid

        hop2_dict[time] = {}
        for k in dd.keys():
            hop2_dict[time][k] = {}
            for u, v in dd[k].items():
                if len(v) >= path_len:
                    hop2_dict[time][k][u] = v

    with open(cache_file, 'wb') as f:
        pickle.dump(hop2_dict, f)
    with open(time_relation_file, 'wb') as f:
        pickle.dump(time_relation, f)
    with open(mid_file, 'wb') as f:
        pickle.dump(mid_dict, f)
    print('The two-hop dict has been saved in', cache_file)
    return hop2_dict,mid_dict,time_relation

def generate_triplet(mid_dict, num_entity,time_list, negative_rate):
    pos_data,pos_label=[[] for i in range(4)],[[] for i in range(4)]
    dir_pair_data,pair_label=[[] for i in range(4)],[[] for i in range(4)]
    for idx,time in enumerate(time_list):
        u_data,mid_data,v_data,label_data=[],[],[],[]
        dir_u_data,dir_mid_data,dir_v_data,dir_label_data=[],[],[],[]
        sam0,sam1=[],[]
        for mid,data in mid_dict[time].items():
            if len(data[0])>1:
                uv=data[0]

                sam0=random.sample(uv,len(uv))
                u_data+=sam0
                mid_data+=[mid]*(len(uv))
                v_data+=uv
                label_data+=[-1]*(len(uv))
            if len(data[1])>1:
                uv=data[1]

                sam1=random.sample(uv,len(uv))
                u_data+=sam1
                mid_data+=[mid]*(len(uv))
                v_data+=uv
                label_data+=[1]*(len(uv))

            dir_pair_len=min(len(data[0]),len(data[1]))
            if dir_pair_len==0: continue

            ds1=random.sample(data[0],len(data[0]))[:dir_pair_len]
            ds2=random.sample(data[1],len(data[1]))[:dir_pair_len]
            dir_u_data+=ds1
            dir_u_data+=ds2
            dir_v_data+=ds2
            dir_v_data+=ds1
            dir_mid_data+=[mid]*(dir_pair_len*2)


        pos_data[idx]=np.stack([u_data,mid_data,v_data],1)
        pos_label[idx]=label_data
        pos_data[idx]=np.tile(pos_data[idx],(5,1))
        pos_label[idx]=np.tile(pos_label[idx],5)

        dir_pair_data[idx]=np.stack([dir_u_data,dir_mid_data,dir_v_data],1)
        pair_label[idx]=np.ones(len(dir_pair_data[idx]))
        pair_label[idx]=np.concatenate((pair_label[idx],np.zeros(len(dir_pair_data[idx]))))
        dir_pair_data[idx]=np.concatenate((dir_pair_data[idx],random.sample(list(pos_data[idx]),len(dir_pair_data[idx]))))
        
    neg_data=[[] for i in range(4)]
    for i in range(4):
        neg_data[i]=negative_sampling_triplet(pos_data[i],pos_label[i],num_entity,negative_rate)
    for i in range(4):
        pair_label[i]=paddle.to_tensor(pair_label[i]).cast('float32')
    return pos_data,neg_data,dir_pair_data,pair_label

def negative_sampling_triplet(pos_samples,pos_label, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = pos_label
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return neg_samples

def generate_sampled_hetero_graphs_and_labels(triplets, num_nodes, relation_dict, time_list, hop2_dict=None, coords=None, test=False, negative_rate=10, split_ratio=0.5,all_grid_dict=None, time_relation=None,far_grid_dict=None,not_evo=None):
    sample_size = len(triplets)
    src, rel, dst = triplets.transpose()
    if not test:
        samples, labels = negative_sampling(triplets, num_nodes, negative_rate)
        split_size = int(sample_size * split_ratio)
        graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size, replace=False)
        src = src[graph_split_ids]
        dst = dst[graph_split_ids]
        rel = rel[graph_split_ids]

    edges_dict = {}
    time_edges_list = []
    mids_dict = {}
    for time in time_list:
        ids = [v for k, v in relation_dict.items() if 'at_' + time in k]
        ids = np.isin(rel, ids)
        src_t = src[ids]
        dst_t = dst[ids]
        time_edges_list.append([(u, v) for u,v in zip(src_t, dst_t)])

        if hop2_dict:
            tmp_g = np.zeros((num_nodes, num_nodes))
            for s, o in zip(src_t, dst_t):
                tmp_g[s][o] = 1
            
            for r, reld in hop2_dict[time].items():
                src_t, dst_t, mid_t = [], [], []
                for (u, v2), vs in reld.items():
                    for v in vs:
                        if tmp_g[u][v] and tmp_g[v][v2]:
                            src_t += [u]
                            dst_t += [v2]
                            mid_t += [v]

                rel_time = r + '_at_' + time
                edges_dict[rel_time] = [(u, v) for u,v in zip(src_t, dst_t)]
                mids_dict[rel_time] = {'ids_1hop': mid_t} 

    tlen = len(time_list)
    for t in range(tlen):
        edges_dict[time_list[t]] = time_edges_list[(t-1)%tlen] + time_edges_list[t] + time_edges_list[(t+1)%tlen]
        mids_dict[time_list[t]] = None

    hg = pgl.HeterGraph(num_nodes=num_nodes,
                       edges=edges_dict,
                       node_feat={'loc': coords}, 
                       edge_feat=mids_dict)

    if test:
        return hg
    else:
        return hg, samples, labels
    
def generate_st_neg_sample(triplets,all_grid_dict=None,time_relation=None,relation_dict=None,grid_sample_size=3,far_grid_dict=None,not_evo=None):
    node2grid={}
    for k,v in all_grid_dict.items():
        for i in v:
            node2grid[i]=k
    src, rel, dst = triplets.transpose()
    all_node_set=set(np.arange(max(src)+1))
    neg_sample=[]

    for i,tri in enumerate(triplets):
        tri_src = src[i]
        tri_dst = dst[i]
        tri_rel = rel[i]
        time_stamp=tri_rel%4
        rel_stamp=int(tri_rel/4)
        
        for node in sample_grid(far_grid_dict[node2grid[tri_src]],grid_sample_size):
            neg_sample.append((tri_src,tri_rel,node))
        for node in sample_grid(far_grid_dict[node2grid[tri_dst]],grid_sample_size):
            neg_sample.append((node,tri_rel,tri_dst))

        neg_sample.append((tri_src,(1-rel_stamp)*4+time_stamp,tri_dst))

    return neg_sample

def sample_grid(lst, k):
    if len(lst) <= k:
        return lst  
    else:
        return random.sample(lst, k) 

def pre_neg_generate(triplets,num_nodes,time_relation,all_grid_dict):
    all_node_set=set(np.arange(num_nodes))
    far_grid_dict={}
    for k,v in all_grid_dict.items():
        far_grid_dict[k]=all_node_set-get_near_grid(k[0],k[1],all_grid_dict)
    
    src, rel, dst = triplets.transpose()
    not_evo=[[] for i in range(4)]
    for i,tri in enumerate(triplets):
        tri_src = src[i]
        tri_dst = dst[i]
        tri_rel = rel[i]
        time_stamp=tri_rel%4
        rel_stamp=int(tri_rel/4)

        if not key_exists_in_nested_dict(time_relation,rel_stamp,(time_stamp+1)%4,tri_src,tri_dst):
            not_evo[(time_stamp+1)%4].append((tri_src,rel_stamp*4+(time_stamp+1)%4,tri_dst))
        if not key_exists_in_nested_dict(time_relation,rel_stamp,(time_stamp-1)%4,tri_src,tri_dst):
            not_evo[(time_stamp-1)%4].append((tri_src,rel_stamp*4+(time_stamp-1)%4,tri_dst))
        
    return far_grid_dict,not_evo

def get_near_grid(grid_x,grid_y,all_grid_dict):
    node_set=set()
    for a in range(-1,2,1):
        for b in range(-1,2,1):
            if (grid_x+a,grid_y+b) not in all_grid_dict:
                continue
            else:
                node_set.update(all_grid_dict[(grid_x+a,grid_y+b)])
    return node_set

def key_exists_in_nested_dict(nested_dict,rel,*keys):
    if not keys:
        return nested_dict==rel
    current_key = keys[0]
    if current_key not in nested_dict:
        return False
    else:
        return key_exists_in_nested_dict(nested_dict[current_key],rel, *keys[1:])

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    samples = np.concatenate((pos_samples, neg_samples))
    _, ids = np.unique(samples, axis=0, return_index=True)
    ids.sort()
    samples = samples[ids]
    labels = labels[ids]
    return samples, labels



def mrr_hit_metric(ranks, hits=[1,2,3,5,10,20]):
    ranks += 1 
    results = {}
    for hit in hits:
        avg_count = np.mean((ranks <= hit))
        results['Hit@'+str(hit)] = avg_count
        if hit >= 1:
            rank_ = 1.0 / ranks
            rank_[rank_ < 1.0/hit] = 0
            results['MRR@'+str(hit)] = np.mean(rank_)

    return results

def overall_mrr_hit_metric(data_dir, dataset, metrics_list, time_list):
    cnt_l = [0 for _ in range(len(time_list))]
    with open('%s/%s/test.txt' % (data_dir, dataset)) as f:
        for line in f.readlines():
            _, rel, _ = line.split('\t')
            for i in range(len(time_list)):
                if '_'+time_list[i] in rel:
                    cnt_l[i] += 1
    overall_metrics = {}
    for key in list(metrics_list[0].keys()):
        scores = [metrics[key] for metrics in metrics_list]
        score = sum([scores[i]*cnt_l[i]/sum(cnt_l) for i in range(len(scores))])
        overall_metrics[key] = score
    return overall_metrics