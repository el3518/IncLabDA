import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth, SupConLoss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import torch.nn.functional as F
from utilities import BCELossForMultiClassification, AccuracyCounter, variable_to_numpy
from torch.autograd import Variable
import loss
import utils_data as com_data
from loss_com import *

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

def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels

def class_prior(complementary_labels):
    return np.bincount(complementary_labels) / len(complementary_labels)

#full_train_loader=source_org, ordinary=source_tr, 
def prepare_train_loaders(full_train_loader, args):
    train_bs = args.batch_size
    for i, (data, labels) in enumerate(full_train_loader):
            K = torch.max(labels)+1 # K is number of classes, full_train_loader is full batch
    complementary_labels = generate_compl_labels(labels)
    ccp = class_prior(complementary_labels)
    complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).float())
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=train_bs, shuffle=True)
    return complementary_train_loader, ccp

    
def data_load_src(args, source_name): 
    ## prepare data
    #source_name = args.s1_dset_path
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(source_name).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()
        
    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    #dsets["source_org"] = ImageList(tr_txt, transform=image_train())
    #dset_loaders["source_org"] = DataLoader(dsets["source_org"], batch_size=len(dsets["source_org"]), shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

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

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

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
        
        idx_known = np.where(matrix.sum(axis=1)>0)
        acc = np.squeeze(matrix[idx_known, idx_known]/np.squeeze(matrix[idx_known, :]).sum(axis=1) * 100) #axis=1 row sum(matrix[1,:])     
        
        
        unknown_acc = acc[-1:].item()
        hm = 2 * (np.mean(acc[:-1]) * unknown_acc)/(np.mean(acc[:-1]) + unknown_acc)
        return np.mean(acc[:-1]), np.mean(acc), unknown_acc, hm
    else:
        return accuracy*100, mean_ent

def cosine_similarity(feature, pairs):
    feature = F.normalize(feature)  # F.normalize只能处理两维的数据，L2归一化
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())  # 计算余弦相似度
    return similarity

def discriminator_loss(net, fea, lab, args):   #lab=labels net=netC
    bin_label = F.one_hot(lab, args.class_num) #fea=netB(images)
    pre_label = net(fea)   
    cls_loss = BCELossForMultiClassification(bin_label, pre_label)
    return cls_loss
    
def proto_generator_loss(args, epoch, generator, netB, netC, netD):
    #generator = generator1
    #netD = netD1
    criterion_g = torch.nn.CrossEntropyLoss()
    gen_c = network.infoNCE_g(class_num=args.class_num)
    z = Variable(torch.rand(args.batch_size*2, 100)).cuda()

    # Get labels ranging from 0 to n_classes for n rows
    labels = Variable(torch.randint(0, args.class_num, (args.batch_size*2,))).cuda()
    z = z.contiguous()
    labels = labels.contiguous()
    images = generator(z, labels)
    #gen_feas = netB(images)
    output_teacher_batch = netC(netB(images))

    # One hot loss
    loss_one_hot = criterion_g(output_teacher_batch, labels)
    loss_binary = discriminator_loss(netD, netB(images), labels, args)

    if epoch >= 30:
    # contrastive loss
        total_contrastive_loss = torch.tensor(0.).cuda()
        contrastive_label = torch.tensor([0]).cuda()

            # MarginNCE
        margin = 0.5
        gamma = 1
        nll = nn.NLLLoss()
        for idx in range(images.size(0)):
            pairs4q = gen_c.get_posAndneg(features=images, labels=labels, feature_q_idx=idx)

            # 余弦相似度 [-1 1]
            result = cosine_similarity(images[idx].unsqueeze(0), pairs4q)

            numerator = torch.exp((result[0][0] - margin) / gamma)
            denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
            # log
            result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
            # nll_loss
            contrastive_loss = nll(result, contrastive_label)

            # contrastive_loss = self.criterion(result, contrastive_label)
            total_contrastive_loss = total_contrastive_loss + contrastive_loss
        total_contrastive_loss = total_contrastive_loss / images.size(0)
    else:
        total_contrastive_loss = torch.tensor(0.).cuda()
 
    return loss_one_hot + total_contrastive_loss + loss_binary

#loader = dset_loaders_s1['source_te']
#proto = norm_fea(torch.from_numpy(proto).float())

def binary_acc(loader, netF, netB, netD, args):
    start_test = True 
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] # ground truth
            inputs = inputs.cuda()
            labels = F.one_hot(labels, args.class_num)
            pred = netD(netB(netF(inputs)))            
            
            if start_test:
                all_pred = pred.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_pred = torch.cat((all_pred, pred.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    #bin_label = F.one_hot(all_label, args.class_num)
    correct = np.equal(np.argmax(variable_to_numpy(all_pred), 1), np.argmax(variable_to_numpy(all_label), 1)).sum()
        
    
    acc = correct / len(all_label)

    return acc * 100

def gen_proto(netB, generator, args):
    z = Variable(torch.rand(args.class_num*2, 100)).cuda()

    # Get labels ranging from 0 to n_classes for n rows
    label_t = torch.linspace(0, args.class_num-1, steps=args.class_num).long()
    for ti in range(args.class_num*2//args.class_num-1):
        label_t = torch.cat([label_t, torch.linspace(0, args.class_num-1, steps=args.class_num).long()])
    labels = Variable(label_t).cuda()
    z = z.contiguous()
    labels = labels.contiguous()
    images = generator(z, labels)
    fea = netB(images)
    
    # obtain prototype of each class
    la_tup = []
    all_class_prototypes = torch.Tensor([]).cuda()
    for i, lab_id in enumerate(labels):
        if lab_id not in la_tup:
            la_tup.append(lab_id)
            all_class_prototypes = torch.cat(
                (all_class_prototypes, fea[i].unsqueeze(0)))

    proto = norm_fea(all_class_prototypes.detach().cpu()).float().numpy()
    return proto

def update_anc_src(aff, all_fea, labelset, K, args):
    
    initc0 = aff.transpose().dot(all_fea)
    initc0 = initc0 / (1e-8 + aff.sum(axis=0)[:,None])  
    
    return initc0

def clu_anc_ini(loader, netF, netB, args):
    start_test = True #loader = dset_loaders_s1["source"]
    with torch.no_grad(): # netB = netB1
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] # ground truth
            inputs = inputs.cuda()
            feas = netB(netF(inputs))

            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                #all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                #all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    
    K = args.class_num#all_output.size(1)
    aff = np.eye(K)[all_label.int()] #label vector sample size classes
        
    cls_count = np.eye(K)[all_label.int()].sum(axis=0) #cluster number 
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    initc = update_anc_src(aff, all_fea, labelset, K, args)

    return initc

def norm_fea(fea):    #fea= torch.from_numpy(proto).float()
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

def proto_acc(loader, netF, netB, args, proto):
    start_test = True 
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] # ground truth
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()  
    
    dd = cdist(all_fea, proto, args.distance)
    clu_label = dd.argmin(axis=1)#.int()
    acc = np.sum(clu_label == all_label.float().numpy()) / len(all_fea)

    return acc * 100

def train_source(args):
    criterion = SupConLoss(temperature=args.temp)
    
    dset_loaders_s1 = data_load_src(args, args.s1_dset_path)
    dset_loaders_s2 = data_load_src(args, args.s2_dset_path)
    dset_loaders_s3 = data_load_src(args, args.s3_dset_path)
    
    '''
    dset_loaders_s1["source_org"], ccp1 = prepare_train_loaders(dset_loaders_s1["source_org"], args)
    dset_loaders_s2["source_org"], ccp2 = prepare_train_loaders(dset_loaders_s2["source_org"], args)
    dset_loaders_s3["source_org"], ccp3 = prepare_train_loaders(dset_loaders_s3["source_org"], args)
    '''
    #dset_loaders_t = data_load_tst(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()   
    
    netD = network.Discriminator(args.class_num).cuda() 
    #netCL = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()   
    
    generator = network.generator_fea_deconv(class_num=args.class_num).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
        
    for k, v in netD.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}] 
    
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    '''
    param_group_c = []
    for k, v in netCL.named_parameters():
        param_group_c += [{'params': v, 'lr': learning_rate}]  
    
    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)
    '''
    
    param_group_g = []
    for k, v in generator.named_parameters():
        param_group_g += [{'params': v, 'lr': learning_rate}] 
    
    optimizer_g = optim.SGD(param_group_g)
    optimizer_g = op_copy(optimizer_g)

    acc_init = 0
    count = 0
    max_iter = args.max_epoch * max(len(dset_loaders_s1["source_tr"]),len(dset_loaders_s2["source_tr"]),len(dset_loaders_s3["source_tr"]))
    interval_iter = max_iter // 10
    iter_num = 0
    
    '''
    cl_method = 'ga'   #choices=['ga', 'nn', 'free', 'pc', 'forward']
    meta_method = 'free' if cl_method =='ga' else cl_method
    K=args.class_num
    '''

    netF.train()
    netB.train()
    netC.train()
    
    netD.train()    
    #netCL.train()
    generator.train()
    
        
    while iter_num < max_iter:
        try:
            inputs_source1, labels_source1 = iter_source1.next()
        except:
            iter_source1 = iter(dset_loaders_s1["source_tr"])
            inputs_source1, labels_source1 = iter_source1.next()

        try:
            inputs_source2, labels_source2 = iter_source2.next()
        except:
            iter_source2 = iter(dset_loaders_s2["source_tr"])
            inputs_source2, labels_source2 = iter_source2.next()
        
        try:
            inputs_source3, labels_source3 = iter_source3.next()
        except:
            iter_source3 = iter(dset_loaders_s3["source_tr"])
            inputs_source3, labels_source3 = iter_source3.next()

        ############################################################
        '''
        try:
            inputs_source1c, labels_source1c = iter_source1c.next()
        except:
            iter_source1c = iter(dset_loaders_s1["source_org"])
            inputs_source1c, labels_source1c = iter_source1c.next()

        try:
            inputs_source2c, labels_source2c = iter_source2c.next()
        except:
            iter_source2c = iter(dset_loaders_s2["source_org"])
            inputs_source2c, labels_source2c = iter_source2c.next()
        
        try:
            inputs_source3c, labels_source3c = iter_source3c.next()
        except:
            iter_source3c = iter(dset_loaders_s3["source_org"])
            inputs_source3c, labels_source3c = iter_source3c.next()

        #######################################################################
        '''
        if inputs_source1.size(0) == 1 or inputs_source2.size(0) == 1 or inputs_source3.size(0) == 1:
            continue

        

        #for idx, (image, labels) in enumerate(dset_loaders_s1["source_tr"]):
        #    print(image.shape)
        #    image[0].shape labels.shape
        #    torch.cat([image, image], dim=0).shape
                
        inputs_source1, labels_source1 = inputs_source1.cuda(), labels_source1.cuda()
        inputs_source2, labels_source2 = inputs_source2.cuda(), labels_source2.cuda()
        inputs_source3, labels_source3 = inputs_source3.cuda(), labels_source3.cuda()
        
        '''
        inputs_source1c, labels_source1c = inputs_source1c.cuda(), labels_source1c.cuda()
        inputs_source2c, labels_source2c = inputs_source2c.cuda(), labels_source2c.cuda()
        inputs_source3c, labels_source3c = inputs_source3c.cuda(), labels_source3c.cuda()
        '''      
        #################################labels.shape labels.T
        
        '''
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            src_cen1 = clu_anc_ini(dset_loaders_s1["source_tr"], netF, netB, args)
            src_cen2 = clu_anc_ini(dset_loaders_s2["source_tr"], netF, netB, args)
            src_cen3 = clu_anc_ini(dset_loaders_s3["source_tr"], netF, netB, args)
            netF.train()
            netB.train()
            netC.train()    
        '''
        
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        optimizer.zero_grad()
        
        #lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)
        #optimizer_c.zero_grad()
        
        classifier_loss = discriminator_loss(netD, netB(netF(inputs_source1)), labels_source1, args)
        #cls_loss = BCELossForMultiClassification(bin_label, pre_label)
               
        outputs_source = netC(netB(netF(inputs_source1)))       
        classifier_loss += CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source1)
         
        src_cen1 = cal_anc_tra(netB(netF(inputs_source1)), outputs_source, args)
        
        classifier_loss.backward() #classifier_
        optimizer.step() 
        #lossa = cls_loss + classifier_loss #+ con_loss
        
        '''
        outputs_sourcec = netCL(netB(netF(inputs_source1c)))
        loss_comp, loss_vector = chosen_loss_c(f=outputs_sourcec, K=K, labels=labels_source1c, ccp=ccp1, meta_method=meta_method)
        #loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        #softmax_output = nn.Softmax(dim=1)(output)
        if cl_method == 'ga':
            if torch.min(loss_vector).item() < 0:
                loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)), 1)
                min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                loss_comp = torch.sum(min_loss_vector)
                loss_comp.backward()
                for group in optimizer.param_groups:
                    for p in group['params']:
                        p.grad = -1*p.grad
            else:
                loss_comp.backward()
        else:
            loss_comp.backward()
        optimizer_c.step()   
        '''

        
        classifier_loss = discriminator_loss(netD, netB(netF(inputs_source2)), labels_source2, args)
        #cls_loss = BCELossForMultiClassification(bin_label, pre_label)
        outputs_source = netC(netB(netF(inputs_source2)))
        classifier_loss += CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source2)  
        
        src_cen2 = cal_anc_tra(netB(netF(inputs_source2)), outputs_source, args)
        
        classifier_loss.backward() #classifier_
        optimizer.step() 
        
        classifier_loss = discriminator_loss(netD, netB(netF(inputs_source3)), labels_source3, args)
        #cls_loss = BCELossForMultiClassification(bin_label, pre_label)
        outputs_source = netC(netB(netF(inputs_source3)))
        classifier_loss += CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source3)  
        
        src_cen3 = cal_anc_tra(netB(netF(inputs_source3)), outputs_source, args)
        
        classifier_loss.backward() #classifier_
        optimizer.step() 
        
        #################################
        lr_scheduler(optimizer_g, iter_num=iter_num, max_iter=max_iter)
        optimizer_g.zero_grad()
        
        generate_loss = proto_generator_loss(args, iter_num, generator, netB, netC, netD)
        
        src_anc = gen_proto(netB, generator, args)                  
        cen_loss = np.linalg.norm(src_anc-src_cen1, ord=2, keepdims=True) 
        cen_loss += np.linalg.norm(src_anc-src_cen2, ord=2, keepdims=True) 
        cen_loss += np.linalg.norm(src_anc-src_cen3, ord=2, keepdims=True)
       
        loss_gen = generate_loss + torch.from_numpy(cen_loss).float().cuda()
        loss_gen.backward() #classifier_
        optimizer_g.step()
 
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            
            netD.eval()
            generator.eval()

            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s1_te, _ = cal_acc(dset_loaders_s1['source_te'], netF, netB, netC, False)
                #proto1 = proto_source(args, netD1)
                acc = binary_acc(dset_loaders_s1['source_te'], netF, netB, netD, args)
                #acc_comp, _ = cal_acc(dset_loaders_s1['source_te'], netF, netB, netCL, False)
                acc_clu = proto_acc(dset_loaders_s1['source_te'], netF, netB, args, src_anc)
                
                log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}, Cluster = {:.2f}%, Bin = {:.2f}%'.format(args.name_src1, iter_num, max_iter, acc_s1_te, acc_clu, acc)
                
                acc_s2_te, _ = cal_acc(dset_loaders_s2['source_te'], netF, netB, netC, False)                
                #proto2 = proto_source(args, netD2)
                acc = binary_acc(dset_loaders_s2['source_te'], netF, netB, netD, args)
                #acc_comp, _ = cal_acc(dset_loaders_s2['source_te'], netF, netB, netCL, False)
                acc_clu = proto_acc(dset_loaders_s2['source_te'], netF, netB, args, src_anc)
                
                
                log_str2 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}, Cluster = {:.2f}%, Bin = {:.2f}%'.format(args.name_src2, iter_num, max_iter, acc_s2_te, acc_clu, acc)
                
                acc_s3_te, _ = cal_acc(dset_loaders_s3['source_te'], netF, netB, netC, False)                
                #proto3 = proto_source(args, netD3)
                acc = binary_acc(dset_loaders_s3['source_te'], netF, netB, netD, args) #norm_fea(torch.from_numpy(proto3).float())
                #acc_comp, _ = cal_acc(dset_loaders_s3['source_te'], netF, netB, netCL, False)
                acc_clu = proto_acc(dset_loaders_s3['source_te'], netF, netB, args, src_anc)
                
                log_str3 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}, Cluster = {:.2f}%, Bin = {:.2f}%'.format(args.name_src3, iter_num, max_iter, acc_s3_te, acc_clu, acc)
            
            
            args.out_file.write(log_str1 + '\n')
            args.out_file.flush()
            print(log_str1+'\n')
            args.out_file.write(log_str2 + '\n')
            args.out_file.flush()
            print(log_str2+'\n')
            args.out_file.write(log_str3 + '\n')
            args.out_file.flush()
            print(log_str3+'\n')

            if acc_s1_te + acc_s2_te + acc_s3_te >= acc_init:
                if acc_s1_te + acc_s2_te + acc_s3_te == acc_init:
                    count += 1
                else:
                    count = 0
                acc_init = acc_s1_te + acc_s2_te + acc_s3_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                
                best_netD = netD.state_dict()
                best_generator = generator.state_dict()
                
                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir_src, args.name_src +"_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir_src, args.name_src +"_C.pt"))  
                
                torch.save(best_netD, osp.join(args.output_dir_src, "_D.pt"))
                torch.save(best_generator, osp.join(args.output_dir_src, "_G.pt"))
                
                #src_cen = gen_proto(netB, generator, args)
           
                np.save(args.output_dir_src+"/"+args.name_src+"_proto.npy", src_anc)
                print('Model saved!')
                test_target(args)
                
                if count > 4:
                    break

            netF.train()
            netB.train()
            netC.train()
            
            netD.train()
            generator.train()

    return netF, netB, netC, netD, generator

def test_target(args):
    #dset_loaders = data_load(args)
    dset_loaders_t = data_load_tst(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    
    args.modelpath = args.output_dir_src + '/' + args.name_src +'_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src +'_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    
    netF.eval()
    netB.eval()
    netC.eval()

    if args.da == 'poda':
        acc_os1, acc_os2, acc_unknown, hm= cal_acc_oda(dset_loaders['test'], netF, netB, netC)
        log_str = 'Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%'.format(args.task, acc_os2, acc_os1, acc_unknown, hm)
    
    else:
        if args.dset=='VISDA-C':
            acc, acc_list = cal_acc(dset_loaders_t['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.task, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders_t['test'], netF, netB, netC, False)                                  
            log_str = '\nTraining: {}, Task: {}, Accuracy_s = {:.2f}%'.format(args.trte, args.task, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s1', type=int, default=0, help="source")
    parser.add_argument('--s2', type=int, default=1, help="source")
    parser.add_argument('--s3', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=3, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='OfficeHome', choices=['OfficeHome', 'offcal'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='poda', choices=['uda', 'pda', 'oda', 'poda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
 
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

    if args.dset == 'OfficeHome':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
        task = [0,1,2,3] 
        #task = [0,1,3,2]
        #task = [0,2,3,1] 
        #task = [1,2,3,0]
    if args.dset == 'offcal':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        #task = [0,1,2,3] 
        #task = [0,1,3,2]
        task = [0,2,3,1] 
        #task = [1,2,3,0]
        args.class_num = 10
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    '''
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    '''
    # torch.backends.cudnn.deterministic = True

    args.sk = 3
    folder = './dataset/'
    args.s1_dset_path = folder + args.dset + '/' + names[task[args.s1]] + 'List.txt'
    args.s2_dset_path = folder + args.dset + '/' + names[task[args.s2]] + 'List.txt'
    args.s3_dset_path = folder + args.dset + '/' + names[task[args.s3]] + 'List.txt'
    args.test_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List.txt'     

    if args.dset == 'OfficeHome':
        if args.da == 'poda':
            args.class_num = 15
            args.class_all = 65
            args.class_share = [i for i in range(10)]
            args.src_private = [i for i in range(10,15)]
            args.tar_private = [i for i in range(15,65)]
            args.src_classes = list(set(args.class_share)|set(args.src_private))
            args.tar_classes = list(set(args.class_share)|set(args.tar_private))
        

    traepo = 'sonly-mix-com-g'
    args.task = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + names[task[args.s3]][0].upper() + '2' + names[task[args.t]][0].upper()
    args.output_dir_src = osp.join(args.output, args.da, args.dset, args.task, traepo)
    args.name_src = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + names[task[args.s3]][0].upper()
    args.name_src1 = names[task[args.s1]][0].upper()
    args.name_src2 = names[task[args.s2]][0].upper()
    args.name_src3 = names[task[args.s3]][0].upper()

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    
    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)
    
