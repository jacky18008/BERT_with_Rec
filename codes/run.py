from sampler import UniformSample_original, shuffle, minibatch
import torch
from transformers import DistilBertTokenizer
from tqdm import tqdm
import utils
import numpy as np
import multiprocessing
from functools import partial

def sampler_setup(dataset, batch_size, device):
    S = UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = S[:, 1]
    negItems = S[:, 2]
    users = users.to(device)
    
    users, posItems, negItems = shuffle(users, posItems, negItems)
    # total_batch = len(users) // batch_size + 1
    return users, posItems, negItems
    
def train(dataset, model, loss_fcn, optimizer, weight_decay, epoch, batch_size=32, device="cpu", neg_k=1):
    # print("start training")
    users, posItems, negItems = sampler_setup(dataset, batch_size, device)
    model.train()
    total_loss = 0.
    total_batch = 0.
    for user_batch, pos_batch, neg_batch in tqdm(minibatch(users, posItems, negItems, batch_size=batch_size)):
        # print("data len: ", len(user_batch), len(pos_batch), len(neg_batch))
        users_emb, pos_emb, neg_emb = model(user_batch, pos_batch, neg_batch)
        loss, reg_loss = loss_fcn(users_emb, pos_emb, neg_emb)
        reg_loss = reg_loss*weight_decay
        loss = loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
        total_batch += 1
    # print("total_batch: ", total_batch)
    return total_loss

def test_one_batch(X, config):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in config["topks"]:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def eval(dataset, model, train_all_pos, config, batch_size=32):
    testDict = dataset.ui_dict
    max_K = max(config["topks"])
    topks = config["topks"]
    multicore = config["multicore_eval"]
    # eval mode with no dropout
    model.eval()
    if config["multicore_eval"] == 1:
        CORES = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    test_one_batch_ = partial(test_one_batch, config=config) # assign config
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // batch_size + 1
        print("start eval ...")
        for batch_users in tqdm(minibatch(users, batch_size=batch_size)):
            # train_all_pos = dataset.getUserPosItems(batch_users)
            allPos = [train_all_pos[user] for user in batch_users]
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config["device"])

            rating = model.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch_, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch_(x))
        scale = float(batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if multicore == 1:
            pool.close()
        print(results)
        return results
    
    
    
    

