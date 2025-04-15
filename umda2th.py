import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network# as network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth, SupConLoss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch.autograd import Variable
from utilities import BCELossForMultiClassification, AccuracyCounter, variable_to_numpy
import mmd

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load_tar(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    #txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load_known(args, know_path): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(know_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last= True)
    #dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    #dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load_tst(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i       
        
        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()
    
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc_oda(loader, netF, netB, netC, flag=False, threshold=0.1):
    start_test = True
    #cal_acc_oda(dset_loaders['test'], netF, netB, netC, True, ENT_THRESHOLD)  
    #loader = dset_loaders_ini['test']
    #flag = True
    #threshold = ENT_THRESHOLD
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))

        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        predict[np.where(labels==iidx)[0]] = args.class_num
            
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        #matrix = matrix[np.unique(predict).astype(int),:]
        #matrix = np.zeros([16,16])
        
        '''
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i 
        class_known = []
        for i in args.class_share:
            class_known.extend([label_map_s[i]])
        
        #class_known = [i for i in range(args.share_num)]
        class_unknown = [matrix.shape[0]-1]
        class_t_list = list(set(class_known) | set(class_unknown))
        matrix = matrix[class_t_list,:]

        #acc = matrix.diagonal()/matrix.sum(axis=1) * 100 #axis=1 row sum(matrix[1,:])   
        acc_known = matrix[class_known, class_known]/matrix[class_known, :].sum(axis=1) * 100 #axis=1 row sum(matrix[1,:])     
        acc_unknown = matrix[matrix.shape[0]-1, matrix.shape[1]-1]/matrix[matrix.shape[0]-1, :].sum() * 100 #axis=1 row sum(matrix[1,:])     
        acc = np.concatenate((acc_known, [acc_unknown]))
        '''
        idx_known = np.where(matrix.sum(axis=1)>0)
        acc = np.squeeze(matrix[idx_known, idx_known]/np.squeeze(matrix[idx_known, :]).sum(axis=1) * 100) #axis=1 row sum(matrix[1,:])     
        
        
        unknown_acc = acc[-1:].item()
        hm = 2 * (np.mean(acc[:-1]) * unknown_acc)/(np.mean(acc[:-1]) + unknown_acc)
        return np.mean(acc[:-1]), np.mean(acc), unknown_acc, hm
    else:
        return accuracy*100, mean_ent

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    probs, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    '''
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    '''
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()

    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    all_fea = all_fea[known_idx,:]
    
    #'''
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    #'''
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()
    
    idex = sel_sam(probs[known_idx].float().numpy(), predict.float().numpy().astype('int'), args)
    
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    '''
    simis = torch.matmul(torch.tensor(proto), torch.tensor(initc).transpose(0,1))
    s_idx = simis.argmax(dim=1)
    t_idx = simis.argmax(dim=0)
    map_s2t =  [(i, s_idx[i].item()) for i in range(len(s_idx))]
    map_t2s =  [(t_idx[i].item(), i) for i in range(len(t_idx))]
    inter = [a for a in map_s2t if a in map_t2s]
    
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    '''
    initc, pred_label_cof = update_cen_lab(all_fea, initc, predict, labelset, K, idex, args)

    idex_same = np.where(pred_label_cof == predict.numpy().astype('int'))[0]
    idex_u = list(set(idex) & set(idex_same))
    
    initc, pred_label = update_cen_lab(all_fea, initc, predict, labelset, K, idex_u, args) 
    
    pred_label[idex_u] = pred_label_cof[idex_u]

    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label

    acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)

    return guess_label.astype('int'), ENT_THRESHOLD, labelset

def sel_sam(prob, lab, args):   
    idex = []
    for cls in range(args.class_num):
        idx = [idx for idx, lab in enumerate(lab) if lab == cls]           
        idxs = [idxs for idxs in idx if prob[idxs] >= np.median(prob[idx])]#-(np.median(prob[idx])-np.min(prob[idx]))/(args.sk+1)]            
        #if len(idx) >= 3:
            #idxs = [idxs for idxs in idx if prob[idxs] >= np.median(prob[idx])]#-(np.median(prob[idx])-np.min(prob[idx]))/(args.sk+1)]
            #idx_sort = np.argsort(-prob[idx])
            #idxs = np.array(idx)[np.array(idx_sort)[0:int(len(idx)/3)]]            
        idex.extend(idxs)    
    return idex

def update_cen_lab(all_fea, initc, predict, labelset, K, idex, args):
            
    confi_pre = predict[idex].float().cpu().int().numpy()
    class_tup = []
    for i in confi_pre:
        if i not in class_tup:
            class_tup.append(i)
    
    aff_confi = np.eye(K)[confi_pre]    
    initc0 = aff_confi.transpose().dot(all_fea[idex])                
    initc0 = initc0 / (1e-8 + aff_confi.sum(axis=0)[:,None])
    for i in range(args.class_num):
        if i not in class_tup:
            initc0[i] = initc[i]
        
    dd = cdist(all_fea[idex], initc0[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea[idex])
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        for i in range(args.class_num):
            if i not in class_tup:
                initc[i] = initc0[i]
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label] 
    
    return initc, pred_label.astype('int')

'''
def weight_contrastive_loss(args, images, labels, t_indx): #tar_idx
    # contrastive loss
    gen_inf = network.infoNCE(class_num=args.class_num)
    
    
    all_sam_indx, all_in, _ = np.intersect1d(t_indx, t_indx, return_indices=True)

    total_contrastive_loss = torch.tensor(0.).cuda()
    contrastive_label = torch.tensor([0]).cuda()

    # MarginNCE
    gamma = 0.07
    nll = nn.NLLLoss()
    if len(all_in) > 0:
        for idx in range(len(all_in)):
            pairs4q = gen_inf.get_posAndneg(features=sor_img_con, labels=labels, tgt_label=tgt_pre_label,
                                                     feature_q_idx=t_indx[all_in[idx]],
                                                     co_fea=all_ref_fea[all_in[idx]].cuda())

            # calculate cosine similarity [-1 1]
            result = cosine_similarity(all_ref_fea[all_in[idx]].unsqueeze(0).cuda(), pairs4q)

            # MarginNCE
            # softmax
            numerator = torch.exp((result[0][0]) / gamma)
            denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
            # log
            result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
            # nll_loss
            contrastive_loss = nll(result, contrastive_label) * sam_confidence[t_indx[all_in[idx]]]
            total_contrastive_loss = total_contrastive_loss + contrastive_loss
        total_contrastive_loss = total_contrastive_loss / len(all_in)

    return total_contrastive_loss
'''

def pred_conf_weight(loader, netF, netB, net, args):
    start_test = True 
    #loader = dset_loaders_ini['test']
    #netD = netD1
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            #labels = data[1] # ground truth
            inputs = inputs.cuda()
            #labels = F.one_hot(labels, args.class_num)
            pred = net(netB(netF(inputs)))            
            
            if start_test:
                all_pred = pred.float().cpu()
                #all_label = labels.float()
                start_test = False
            else:
                all_pred = torch.cat((all_pred, pred.float().cpu()), 0)
                #all_label = torch.cat((all_label, labels.float().cpu()), 0)

    all_pred = nn.Softmax(dim=1)(all_pred)
    return all_pred

def pred_list_km(all_pred):    
    probs, pred_lab = torch.max(all_pred, 1)
    
    ent = torch.sum(-all_pred * torch.log(all_pred + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()

    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0] 
    
    all_idx = [i for i in range(len(pred_lab))]
    unknown_idx = list(set(all_idx) - set(known_idx))
    
    return known_idx, unknown_idx, probs, pred_lab

#'''
def file_known_list(known_idx, unknown_idx, pred_lab, args, probs):
    org_file = args.t_dset_path
    file_org = open(org_file, 'r').readlines()
    
    out_file = args.k_tar_path
    file = open(out_file,'w')
    idx_sort = np.argsort(-probs[known_idx])
    idxs = np.array(known_idx)[np.array(idx_sort)[0:int(len(known_idx)/3)]]                                   
    for i in idxs:
        lines = file_org[i]
        line = lines.strip().split(' ')
        new_lines = line[0]
        file.write('%s %s\n' % (new_lines, int(pred_lab[i])))
        #file.write(lines)
    file.close()
    sam_num_k = len(idxs)
    out_file = args.uk_tar_path
    file = open(out_file,'w') 
    idx_sort = np.argsort(probs[unknown_idx])
    idxs = np.array(unknown_idx)[np.array(idx_sort)[0:int(len(unknown_idx)/3)]]            
    for i in idxs:
        lines = file_org[i]
        line = lines.strip().split(' ')
        new_lines = line[0]
        file.write('%s %s\n' % (new_lines, int(pred_lab[i])))
        #file.write(lines)
    file.close()
    sam_num_uk = len(idxs)
    return sam_num_k, sam_num_uk
'''
def file_known_list(known_idx, unknown_idx, pred_lab, args, gap, probs):
    org_file = args.t_dset_path
    file_org = open(org_file, 'r').readlines()
    
    out_file = args.k_tar_path
    file = open(out_file,'w')                   
    for i in known_idx:
        lines = file_org[i]
        line = lines.strip().split(' ')
        new_lines = line[0]
        file.write('%s %s\n' % (new_lines, int(pred_lab[i])))
        #file.write(lines)    
    idx_sort = np.argsort(probs[unknown_idx])
    idxs = np.array(unknown_idx)[np.array(idx_sort)[0:int(len(unknown_idx)/gap)]]            
    for i in idxs:
        lines = file_org[i]
        line = lines.strip().split(' ')
        new_lines = line[0]
        #file.write(lines)
        file.write('%s %s\n' % (new_lines, int(args.class_num)))
    file.close()
'''    

def norm_fea(fea):    
    fea = torch.cat((fea, torch.ones(fea.size(0), 1)), 1)
    fea = (fea.t() / torch.norm(fea, p=2, dim=1)).t()   
    return fea

def cal_anc_tra(fea, output, args):
    
    #cen = np.zeros([])
    fea = fea.detach().cpu()
    if args.distance == 'cosine':
        fea = norm_fea(fea).float().numpy()
    output = nn.Softmax(dim=1)(output)
    aff = output.detach().float().cpu().numpy()
    initc = aff.transpose().dot(fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            	   
    return initc

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(args):
    dset_loaders_ini = data_load_tar(args)
    criterion = SupConLoss(temperature=args.temp)
    #dset_loaders_te = data_load_tar(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netD = network.Discriminator(args.class_num).cuda()  
    #netD2 = network.Discriminator(args.class_num).cuda() 
    #netD3 = network.Discriminator(args.class_num).cuda()  
    
    #netK = network.MDAdomnet(2).cuda()   
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src +'_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src + '_C.pt'    
    netC.load_state_dict(torch.load(args.modelpath))

    args.modelpath = args.output_dir_src + '/'  + '_D.pt'   
    netD.load_state_dict(torch.load(args.modelpath))
    '''
    args.modelpath = args.output_dir_src + '/'  + '_D2.pt'    
    netD2.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/'  + '_D3.pt'    
    netD3.load_state_dict(torch.load(args.modelpath))
    '''
        
    netC.eval()
    netD.eval()
    #netD2.eval()
    #netD3.eval()
            
    for k, v in netC.named_parameters():
        v.requires_grad = False   
    for k, v in netD.named_parameters():
        v.requires_grad = False
    '''
    for k, v in netD2.named_parameters():
        v.requires_grad = False
    for k, v in netD3.named_parameters():
        v.requires_grad = False
    '''
    
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    '''
    for k, v in netK.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    '''
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    acc_init = 0
    tt = 0
    iter_num = 0
    max_iter = args.max_epoch * len(dset_loaders_ini["target"])
    interval_iter = max_iter // args.interval

    src_anc = np.load(args.output_dir_src+"/"+args.name_src+"_proto.npy")
    #src_anc2 = np.load(args.output_dir_src+"/"+args.name_src+"_proto2.npy")
    #src_anc3 = np.load(args.output_dir_src+"/"+args.name_src+"_proto3.npy")
    
    
    while iter_num < max_iter:
        optimizer.zero_grad()
        #'''
        if iter_num  % interval_iter == 0:
            netF.eval()
            netB.eval()
            pred = pred_conf_weight(dset_loaders_ini['test'], netF, netB, netD, args)
            #pred2 = pred_conf_weight(dset_loaders_ini['test'], netF, netB, netD2, args)
            #pred3 = pred_conf_weight(dset_loaders_ini['test'], netF, netB, netD3, args)
            #pred = (pred1 + pred2 + pred3)/3
            known_idx_d, unknown_idx_d, probs_d, _ = pred_list_km(pred)
            pred = pred_conf_weight(dset_loaders_ini['test'], netF, netB, netC, args)
            known_idx, unknown_idx, probs_c, pred_lab = pred_list_km(pred)
            known_idx = list(set(known_idx_d) & set(known_idx))
            unknown_idx = list(set(unknown_idx_d) & set(unknown_idx))
            
            sam_num_k, sam_num_uk = file_known_list(known_idx, unknown_idx, pred_lab, args, probs_c + probs_d)
            s_num = np.array([sam_num_k, sam_num_uk])
            effective_num = 1.0 - np.power(args.beta, s_num)
            weights = (1.0 - args.beta) / np.array(effective_num)
            weights = weights / np.sum(weights)# * s_num
            
            netF.train()
            netB.train()
            dset_loaders_k = data_load_known(args, args.k_tar_path)
            dset_loaders_uk = data_load_known(args, args.uk_tar_path)
                    
        try:
            inputs_k, target_k, _ = iter_test_k.next()
        except:
            iter_test_k = iter(dset_loaders_k["target"])
            inputs_k, target_k, _ = iter_test_k.next()
        
        #'''
        try:
            inputs_uk, target_uk, _ = iter_test_uk.next()
        except:
            iter_test_uk = iter(dset_loaders_uk["target"])
            inputs_uk, target_uk, _ = iter_test_uk.next()
        #'''
        if inputs_k.size(0) == 1 or inputs_uk.size(0) == 1:
            continue 
     
        inputs_k, target_k = inputs_k.cuda(), target_k.cuda()
        inputs_uk, target_uk = inputs_uk.cuda(), target_uk.cuda()
        
        '''
        pred_k = netK(netB(netF(inputs_k)))#weights[0] * 
        k_label = torch.zeros(inputs_k.shape[0]).long().cuda()
        k_loss = F.nll_loss(F.log_softmax(pred_k,dim=1), k_label)
        pred_k = netK(netB(netF(inputs_uk)))
        k_label = torch.ones(inputs_uk.shape[0]).long().cuda()
        k_loss += F.nll_loss(F.log_softmax(pred_k,dim=1), k_label)
        '''       
        uk_loss = -mmd.mmd(netB(netF(inputs_k)), netB(netF(inputs_uk)))         
        
        #'''#inputs_k = inputs_test
        bsz = target_k.shape[0]
        images = torch.cat([inputs_k, inputs_k], dim=0)
        features = F.normalize(netB(netF(images)), dim=1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        con_loss = criterion(features, target_k)
        #con_loss.backward()
        #optimizer.step()
        #'''
        #'''
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders_ini["target"])
            inputs_test, _, tar_idx = iter_test.next()
                
        if inputs_test.size(0) == 1:# or inputs_k.size(0) == 1:
            continue                
        
        inputs_test = inputs_test.cuda()               
                                
        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            mem_label, ENT_THRESHOLD, labelset = obtain_label(dset_loaders_ini['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()
        
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        pred = mem_label[tar_idx]
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        outputs_test_known = outputs_test[pred < args.class_num, :]
        features_test_known = features_test[pred < args.class_num, :] 
        pred = pred[pred < args.class_num] 
               
        if len(pred != 0):
                        
            tar_cen = cal_anc_tra(features_test_known, outputs_test_known, args)
            
            cen_loss = np.linalg.norm(src_anc[labelset]-tar_cen[labelset], ord=2, keepdims=True) 
            #cen_loss2 = np.linalg.norm(src_anc2[labelset]-tar_cen[labelset], ord=2, keepdims=True) 
            #cen_loss3 = np.linalg.norm(src_anc3[labelset]-tar_cen[labelset], ord=2, keepdims=True)
            
            cen_loss = torch.from_numpy(cen_loss).float().cuda()
        else:
            cen_loss = torch.tensor([0]).cuda()
        
        if len(pred) == 0:
            print(tt)
            del features_test
            del outputs_test
            tt += 1
            continue        
        
        if args.cls_par > 0:
            classifier_loss = nn.CrossEntropyLoss()(outputs_test_known, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out_known = nn.Softmax(dim=1)(outputs_test_known)
            entropy_loss = torch.mean(loss.Entropy(softmax_out_known))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            classifier_loss += entropy_loss * args.ent_par

        total_loss = classifier_loss + con_loss + uk_loss + cen_loss #+ k_loss 
        #classifier_loss += con_loss 
        total_loss.backward()
        optimizer.step()
                           
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc_os1, acc_os2, acc_unknown, hm = cal_acc_oda(dset_loaders_ini['test'], netF, netB, netC, True, ENT_THRESHOLD)            
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%'.format(args.task, iter_num, max_iter, acc_os2, acc_os1, acc_unknown, hm)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            if acc_os2  >= acc_init:
                acc_init = acc_os2 
                if args.issave:   
                    torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                    torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
                    torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

            print(log_str+'\n')
            netF.train()
            netB.train()
        
    return netF, netB, netC

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s1', type=int, default=0, help="source")
    parser.add_argument('--s2', type=int, default=1, help="source")
    parser.add_argument('--t', type=int, default=2, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office31', choices=['VISDA-C', 'office31', 'OfficeHome', 'offcal', 'CLEF', 'DomainNet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--da', type=str, default='poda', choices=['uda', 'pda', 'oda', 'poda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--issave', type=bool, default=True)#True
    
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
       
    #  other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    
    args = parser.parse_args()
       
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        #task = [0,1,2] 
        #task = [2,1,0]
        task = [0,2,1]
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    '''
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    '''
    # torch.backends.cudnn.deterministic = True
    args.beta = 0.9999
    
    args.sk = 2
    folder = './dataset/'
    args.s1_dset_path = folder + args.dset + '/' + names[task[args.s1]] + 'List.txt'
    args.s2_dset_path = folder + args.dset + '/' + names[task[args.s2]] + 'List.txt'
  
    args.t_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List.txt'
    args.k_tar_path = folder + args.dset + '/' + names[task[args.t]] + 'known.txt'
    args.uk_tar_path = folder + args.dset + '/' + names[task[args.t]] + 'unknown.txt'
    
    if args.dset == 'office31':
        if args.da == 'poda':
            args.class_num = 20
            args.class_all = 31           
            args.class_share1 = [0,1,5,10,11,12,15,16]         
            args.class_share2 = [5,10,11,12,15,16,17,22]
            args.class_share = list(set(args.class_share1)|set(args.class_share2))
            args.src_private1 = [2,3,4,6,7]
            args.src_private2 = [8,9,13,14,18]
            args.tar_private = [19,20,21,23,24,25,26,27,28,29,30]

            args.src_classes = list(set(args.class_share)|set(args.src_private1)|set(args.src_private2))
            args.src_classes1 = list(set(args.class_share1)|set(args.src_private1))
            args.src_classes2 = list(set(args.class_share2)|set(args.src_private2))
            args.tar_classes = list(set(args.class_share)|set(args.tar_private))
    
    load_tag = 'sonly-mix-comhh'
    args.name_src = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper()
    args.task = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + '2' + names[task[args.t]][0].upper()   
    args.output_dir_src = osp.join(args.output, args.da, args.dset, args.task, load_tag)
    traepo = 2
    save_tag = 'epo_tar-comhh' + str(traepo)
    args.output_dir = osp.join(args.output, args.da, args.dset, args.task, save_tag)

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)
    if args.da == 'pda':
        args.gent = ''
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)

