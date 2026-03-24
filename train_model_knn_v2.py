from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time
import copy
from evaluate import fx_calc_map_label, fx_calc_map_multilabel
import numpy as np
from matplotlib import pyplot as plt
from losses import SupConLoss
import scipy.io as sio
import scipy
import scipy.spatial
from sklearn.metrics import accuracy_score
import ot
import random
from scipy.spatial.distance import cdist
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
from load_data import CustomDataSet
def rank_loss(features1, features2, margin):
    sim12 = features1.mm(features2.t())
    diag = torch.diag(sim12)
    sim12 = sim12 - diag.view(-1,1) + margin
    sim12[sim12 < 0] = 0

    sim21 = features2.mm(features1.t())
    diag = torch.diag(sim21)
    sim21 = sim21 - diag.view(-1,1) + margin
    sim21[sim21 < 0] = 0
    return sim12.mean() + sim21.mean()

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t()) 
    return Sim


def train_model_synchronous(model, input_data_par, optimizer, args):
    img_train, txt_train, label_train_ori, label_train_noisy = input_data_par['img_train'], input_data_par['text_train'], input_data_par['label_train_ori'], input_data_par['label_train_noisy']
    img_valid, txt_valid, label_valid = input_data_par['img_valid'], input_data_par['text_valid'], input_data_par['label_valid']
    train_clean_indices = np.argmax(label_train_noisy, axis=1) == np.argmax(label_train_ori, axis=1)
    train_clean_indices = np.where(train_clean_indices)[0]
    num_epochs = args.MAX_EPOCH
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    MAPI2T_list, MAPT2I_list, Clean_num_selected_list, num_selected_list ,Clean_num_all_list = [], [], [], [], []

    train_dataset = CustomDataSet(img_train, txt_train, label_train_noisy, label_train_ori)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = CustomDataSet(img_valid, txt_valid, label_valid, label_valid)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        barycenters = get_barycenters(model, train_loader, args)
        barycenters = torch.tensor(barycenters, requires_grad=False).cuda()
        pure_clean_ids, hard_ids, noisy_ids, img_soft_labels_reordered, txt_soft_labels_reordered, selected_clean_num, num_selected, all_clean_num = divide_sample(model, train_loader, args)
        img_soft_labels_reordered = img_soft_labels_reordered.detach().cuda()
        txt_soft_labels_reordered = txt_soft_labels_reordered.detach().cuda()
        Clean_num_selected_list.append(selected_clean_num)
        num_selected_list.append(num_selected)
        Clean_num_all_list.append(all_clean_num)

        pure_dataset = CustomDataSet(img_train[pure_clean_ids], txt_train[pure_clean_ids], label_train_noisy[pure_clean_ids], label_train_ori[pure_clean_ids])
        pure_loader = torch.utils.data.DataLoader(pure_dataset, batch_size=args.batch_size, shuffle=True)
        train_pure(model, pure_loader, optimizer, barycenters, args)
        if len(hard_ids) > 1 and args.hard_train:
            print(len(hard_ids))
            hard_dataset = CustomDataSet(img_train[hard_ids], txt_train[hard_ids], label_train_noisy[hard_ids], label_train_ori[hard_ids])
            hard_loader = torch.utils.data.DataLoader(hard_dataset, batch_size=args.batch_size, shuffle=True)
            train_hard(model, hard_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args)
        if len(noisy_ids) > 1 and args.noisy_train:
            noisy_dataset = CustomDataSet(img_train[noisy_ids], txt_train[noisy_ids], label_train_noisy[noisy_ids], label_train_ori[noisy_ids])
            noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=args.batch_size, shuffle=True)
            train_noisy(model, noisy_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args)

        model.eval()
        t_imgs_fea, t_imgs_pred, t_txts_fea, t_txts_pred, t_labels = [], [], [], [], []
        with torch.no_grad():
            for imgs, txts, labels_noisy, labels_ori, index in valid_loader:
                if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels_ori.cuda()
                t_view1_feature, t_view2_feature = model(imgs, txts)
                t_view1_predict = F.softmax(t_view1_feature.view([t_view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
                t_view2_predict = F.softmax(t_view2_feature.view([t_view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
                t_imgs_fea.append(t_view1_feature.cpu().numpy())
                t_imgs_pred.append(t_view1_predict.cpu().numpy())
                t_txts_fea.append(t_view2_feature.cpu().numpy())
                t_txts_pred.append(t_view2_predict.cpu().numpy())
                t_labels.append(labels.cpu().numpy())
        t_imgs_fea = np.concatenate(t_imgs_fea)
        t_imgs_pred = np.concatenate(t_imgs_pred)
        t_txts_fea = np.concatenate(t_txts_fea)
        t_txts_pred = np.concatenate(t_txts_pred)
        t_labels = np.concatenate(t_labels)
        img2txt = fx_calc_map_multilabel(t_imgs_fea, t_txts_fea, t_labels, metric='cosine')
        txt2img = fx_calc_map_multilabel(t_txts_fea, t_imgs_fea, t_labels, metric='cosine')
        MAPI2T_list.append(img2txt)
        MAPT2I_list.append(txt2img)
        num_val = t_labels.shape[0]
        img_acc = np.sum(np.argmax(t_imgs_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val
        txt_acc = np.sum(np.argmax(t_txts_pred, axis=1) == np.argmax(t_labels, axis=1)) / num_val

        print('Img2Txt: %.4f  Txt2Img: %.4f Imgacc: %.4f  Txtacc: %.4f lr: %g'%(img2txt, txt2img, img_acc, txt_acc, optimizer.param_groups[0]['lr']))
        if (img2txt + txt2img) / 2 > best_acc:
            best_acc = (img2txt + txt2img) / 2
            best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, MAPI2T_list, MAPT2I_list, Clean_num_selected_list, num_selected_list, Clean_num_all_list

def train_pure(model, train_loader, optimizer, barycenters, args):
    model.train()
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        tmp1 = - (labels * view1_predict.log()).sum(1)
        tmp2 = - (labels * view2_predict.log()).sum(1)
        term1 = tmp1 + tmp2
        term2 = rank_loss(view1_feature, view2_feature, args.margin)
        loss = args.lamda * term1.mean() + args.alpha * term2
        loss.backward()
        optimizer.step()
def train_hard(model, train_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args):
    model.train()
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view1_weight = view1_predict.clone().detach()
        view2_weight = view2_predict.clone().detach()
        weight = 1 - (1 - view1_weight) * (1 - view2_weight)
        if args.hard_weight:
            tmp1 = - (weight * labels * view1_predict.log()).sum(1)
            tmp2 = - (weight * labels * view2_predict.log()).sum(1)
            term1 = tmp1 + tmp2
        else:
            tmp1 = - (labels * view1_predict.log()).sum(1)
            tmp2 = - (labels * view2_predict.log()).sum(1)
            term1 = tmp1 + tmp2
        term2 = rank_loss(view1_feature, view2_feature, args.margin)
        loss = args.lamda * term1.mean() + args.alpha * term2
        loss.backward()
        optimizer.step()
def train_noisy(model, train_loader, optimizer, barycenters, img_soft_labels_reordered, txt_soft_labels_reordered, args):
    model.train()
    all_pseudo_indices, all_ori_labels = [], []
    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
            print("Data contains Nan.")
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels_noisy.cuda()
                labels_ori = labels_ori.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(barycenters.T), dim=1)
        num_classes = view1_predict.shape[1]

        soft_labels = 1 - (1- view1_predict) * (1 - view2_predict)
        pesdo_labels_idx = torch.argmax(soft_labels, dim=1).detach()
        pesdo_labels = F.one_hot(pesdo_labels_idx, num_classes=num_classes).float()
        all_pseudo_indices.extend(pesdo_labels_idx.cpu().numpy())
        all_ori_labels.extend(torch.argmax(labels_ori, dim=1).cpu().numpy())

        tmp1 = (pesdo_labels - view1_predict).abs().sum(1)
        tmp2 = (pesdo_labels - view2_predict).abs().sum(1)
        term1 = tmp1 + tmp2
        term2 = rank_loss(view1_feature, view2_feature, args.margin)
        loss = args.lamda * term1.mean() + args.alpha * term2
        loss.backward()
        optimizer.step()
def divide_sample(model, train_loader, args):
    model.eval()
    t_imgs_fea, t_txts_fea, t_labels, t_labels_ori, sample_ids = [], [], [], [], []
    clean_indexs, noisy_indexs = [], []

    with torch.no_grad():
        for imgs, txts, labels_noisy, labels_ori, index in train_loader:
            clean_index = torch.argmax(labels_noisy, dim=1) == torch.argmax(labels_ori, dim=1)
            noisy_index = ~clean_index
            clean_indexs.append(index[clean_index])
            noisy_indexs.append(index[noisy_index])

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels_noisy = labels_noisy.cuda()

            t_view1_feature, t_view2_feature = model(imgs, txts)
            t_imgs_fea.append(t_view1_feature.cpu())
            t_txts_fea.append(t_view2_feature.cpu())
            t_labels.append(labels_noisy.cpu())
            t_labels_ori.append(labels_ori.cpu())
            sample_ids.append(index)

    t_imgs_fea = torch.cat(t_imgs_fea, dim=0)
    t_txts_fea = torch.cat(t_txts_fea, dim=0)
    t_labels = torch.cat(t_labels, dim=0)
    t_labels_ori = torch.cat(t_labels_ori, dim=0)
    sample_ids = torch.cat(sample_ids, dim=0)
    clean_indexs = torch.cat(clean_indexs, dim=0)
    noisy_indexs = torch.cat(noisy_indexs, dim=0)

    img_soft_labels = get_soft_labels(t_imgs_fea, t_labels, args.top_k)
    txt_soft_labels = get_soft_labels(t_txts_fea, t_labels, args.top_k)

    img_preds = torch.argmax(img_soft_labels, dim=1)
    txt_preds = torch.argmax(txt_soft_labels, dim=1)
    true_labels = torch.argmax(t_labels, dim=1)

    img_clean = (img_preds == true_labels)
    txt_clean = (txt_preds == true_labels)

    pure_clean_mask = img_clean & txt_clean
    hard_mask = img_clean ^ txt_clean
    noisy_mask = ~(img_clean | txt_clean)

    pure_clean_ids = sample_ids[pure_clean_mask]
    hard_ids = sample_ids[hard_mask]
    noisy_ids = sample_ids[noisy_mask]

    def compute_selection_accuracy(pred_ids, clean_indexs):
        pred_set = set(pred_ids.cpu().numpy().tolist())
        real_clean_set = set(clean_indexs.cpu().numpy().tolist())
        correct = len(pred_set & real_clean_set)
        accuracy = correct / len(pred_set) if len(pred_set) > 0 else 0.0
        return accuracy, correct

    clean_acc, clean_num = compute_selection_accuracy(pure_clean_ids, clean_indexs)
    hard_acc, hard_num = compute_selection_accuracy(hard_ids, clean_indexs)
    img_soft_labels_reordered = torch.zeros_like(img_soft_labels)
    txt_soft_labels_reordered = torch.zeros_like(txt_soft_labels)
    img_soft_labels_reordered[sample_ids] = img_soft_labels
    txt_soft_labels_reordered[sample_ids] = txt_soft_labels
    return pure_clean_ids, hard_ids, noisy_ids, img_soft_labels_reordered, txt_soft_labels_reordered, clean_num, pure_clean_ids.shape[0], clean_indexs.shape[0]


def get_barycenters(model, train_loader, args):
    # 计算每个类别的特征质心
    model.eval()
    train_feature_view1, train_feature_view2 = [], []
    train_label = np.array([]).astype('int16')

    for imgs, txts, labels_noisy, labels_ori, index in train_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            txts = txts.cuda()
            labels = labels_noisy.cuda()
        view1_feature, view2_feature = model(imgs, txts)
        train_feature_view1.append(view1_feature.cpu().detach().numpy())
        train_feature_view2.append(view2_feature.cpu().detach().numpy())
        train_label = np.concatenate((train_label, np.argmax(labels.cpu().detach().numpy(), axis=1)))
    train_feature_view1 = np.concatenate(train_feature_view1, axis=0)
    train_feature_view2 = np.concatenate(train_feature_view2, axis=0)
    

    barycenters = []
    for class_id in range(labels.shape[1]):
        sample = np.where(train_label==class_id)
        sample_feature_view1 = train_feature_view1[sample]
        sample_feature_view2 = train_feature_view2[sample]
        sample_feature = np.concatenate((sample_feature_view1, sample_feature_view2), axis=0)
        center = k_barycenter(sample_feature.transpose(), args.barycenter_number, args.lambd)
        barycenters += center.transpose().tolist()
    barycenters = np.array(barycenters)
    return barycenters.astype('float32')

def k_barycenter(Q, k, lambd):
    c = len(Q)
    m = len(Q[0])

    H = np.zeros((c,k))
    for i in range(k):
        point = random.randint(0,m-1)
        for j in range(c):
            H[j,i] = Q[j,point]
    
    t = 1.1
    eta = 0.5
    b = np.ones(m)/m

    a1 = np.ones(k)/k
    a2 = np.ones(k)/k
    
    a1_former = np.zeros(k)
    H_former = np.zeros((c,k))
    
    convergence_a1=pow(10,-2)
    convergence_H=pow(10,-2)

    while np.linalg.norm(H-H_former)>convergence_H:

        MHQ = cdist(H.transpose(), Q.transpose(), metric='euclidean')
        a1 = np.ones(k)/k
        a2 = np.ones(k)/k
        a1_former = np.zeros(k)
        while np.linalg.norm(a1-a1_former)>convergence_a1:
            beta = (t+1)/2
            a = (1-1/beta)*a1 + (1/beta)*a2
            result, dual = ot.sinkhorn(a, b, MHQ, lambd, verbose=False, log=True)
            alpha = dual['u']
            alpha = (-beta)*alpha
            alpha = np.exp(alpha)
            
            a2 = a2
            a2_n = a2*alpha
            
            if np.sum(np.isinf(a2_n))==1:
                a2 = np.zeros((len(a2),))
                a2[np.isinf(a2_n)]=1
            elif np.all(a2_n==0):
                a2 = np.ones((len(a2),))/len(a2)
            else:
                a2 = a2_n/np.sum(a2_n)
            
            a1_former = a1
            a1 = (1-1/beta)*a1 + (1/beta)*a2
            t+=1
       
        a = a1
        T = ot.sinkhorn(a, b, MHQ, lambd, verbose=False)
        T = T.transpose()
        diag_a_reverse = np.diag(1/a)
        H_former = H
        H = (1-eta)*H + eta*np.dot(np.dot(Q,T),diag_a_reverse)
    
    return H

def get_soft_labels(features: torch.Tensor, labels: torch.Tensor, top_k: int = 10):
    """
    基于余弦相似度使用 PyTorch 实现的软标签生成函数。
    参数:
        features (Tensor): [N, D] 特征张量，float32
        labels   (Tensor): [N, C] one-hot 标签，float32 或 int
        top_k    (int):    每个样本选取的最近邻个数
    返回:
        soft_labels (Tensor): [N, C] 每个样本的软标签（邻居标签平均）
    """
    
    # 计算余弦相似度矩阵 [N, N]
    sim_matrix = torch.matmul(features, features.T)  # 内积即余弦相似度

    # 排除自身相似度
    N = sim_matrix.shape[0]
    sim_matrix.fill_diagonal_(-1)  # 设置对角线为 -1，排除自身

    # 获取 top_k 个最大相似度的邻居索引
    topk_values, topk_indices = torch.topk(sim_matrix, top_k, dim=1)

    # 利用邻居索引获取邻居标签 [N, top_k, C]
    neighbor_labels = labels[topk_indices]  # 索引是 [N, top_k]，labels 是 [N, C] → 输出 [N, top_k, C]

    # 计算平均标签（作为 soft label）[N, C]
    soft_labels = neighbor_labels.float().mean(dim=1)

    return soft_labels
