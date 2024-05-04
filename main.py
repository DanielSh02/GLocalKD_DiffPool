import numpy as np
from sklearn.metrics import auc, roc_curve
import argparse
import load_data
import networkx as nx
import torch
import torch.nn as nn
import time
import gcn
from torch.autograd import Variable
from graph_sampler import GraphSampler
from numpy.random import seed
import random
import torch.nn.functional as F
from plot import plot
from itertools import product

def arg_parse():
    parser = argparse.ArgumentParser(description='GLocalKD Arguments.')
    parser.add_argument('--datadir', dest='datadir', default ='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default ='Tox21_MMP', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=150, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=2000, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=512, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=256, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=3, type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias. Default to True.')
    parser.add_argument('--feature', dest='feature', default='deg-num', help='use what node feature')
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='seed')
    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train(dataset, data_test_loader, model_teacher, model_student, num_epochs, lr, args):    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model_student.parameters()), lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=50, gamma=0.5)
    epochs=[]
    auroc_final = 0
    test_roc_abs = []
    time_0 = time.time()
    times = []
    for epoch in range(num_epochs):
        # print(f"Epoch: {epoch}")
        total_time = 0
        total_loss = 0.0
        model_student.train()

        for batch_idx, data in enumerate(dataset):           
            begin_time = time.time()
            model_student.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False)
            h0 = Variable(data['feats'].float(), requires_grad=False)
           
            embed, lp_loss, entropy_loss = model_student(h0, adj)
            embed_teacher, _, _ = model_teacher(h0, adj)
            embed_teacher =  embed_teacher.detach()
            loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0)
            loss = loss + lp_loss + entropy_loss
            
            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_student.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
                   
        if (epoch+1)%1 == 0 and epoch > 0:
            epochs.append(epoch)
            model_student.eval()   
            loss = []
            y=[]
            emb=[]
            
            for batch_idx, data in enumerate(data_test_loader):
               adj = Variable(data['adj'].float(), requires_grad=False)
               h0 = Variable(data['feats'].float(), requires_grad=False)
                        
               embed, lp_loss, entropy_loss = model_student(h0, adj)
               embed_teacher, _, _ = model_teacher(h0, adj)
               loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1)
               loss_ = loss_graph
               loss_ = np.array(loss_.cpu().detach())
               loss.append(loss_)
               y.append(data['label'])           
               emb.append(embed.cpu().detach().numpy())
            
            label_test = []
            for loss_ in loss:
               label_test.append(loss_)
            label_test = np.array(label_test)
            y_array = np.array([tensor.item() for tensor in y])
            fpr_ab, tpr_ab, _ = roc_curve(y_array, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)   
            # print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
            test_roc_abs.append((test_roc_ab, epoch))
            times.append(time.time() - time_0)
        if (epoch)%25 == 0:
            print(f"Epoch {epoch} complete.")
        if epoch == (num_epochs-1):
            auroc_final =  test_roc_ab
            print(auroc_final)
            # print(sorted(test_roc_abs, key=lambda x: x[0], reverse=True))

    return test_roc_abs, times

def train_regular(dataset, data_test_loader, model_teacher, model_student, num_epochs, lr, args):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model_student.parameters()), lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=50, gamma=0.5)
    epochs=[]
    times = []
    test_roc_abs = []
    auroc_final = 0
    time_0 = time.time()
    for epoch in range(num_epochs):
        # print(f"Epoch: {epoch}")
        total_time = 0
        total_loss = 0.0
        model_student.train()

        for batch_idx, data in enumerate(dataset):           
            begin_time = time.time()
            model_student.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False)
            h0 = Variable(data['feats'].float(), requires_grad=False)
           
            embed_node, embed = model_student(h0, adj)
            embed_teacher_node, embed_teacher = model_teacher(h0, adj)
            embed_teacher =  embed_teacher.detach()
            embed_teacher_node = embed_teacher_node.detach()
            loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0)
            loss = loss + loss_node
            
            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_student.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed

        if (epoch+1)%1 == 0 and epoch > 0:
            epochs.append(epoch)
            model_student.eval()   
            loss = []
            y=[]
            emb=[]
            
            for batch_idx, data in enumerate(data_test_loader):
               adj = Variable(data['adj'].float(), requires_grad=False)
               h0 = Variable(data['feats'].float(), requires_grad=False)
                        
               embed_node, embed = model_student(h0, adj)
               embed_teacher_node, embed_teacher = model_teacher(h0, adj)
               loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1)
               loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1)
               loss_ = loss_graph + loss_node
               loss_ = np.array(loss_.cpu().detach())
               loss.append(loss_)
               y.append(data['label'])           
               emb.append(embed.cpu().detach().numpy())
            
            label_test = []
            for loss_ in loss:
               label_test.append(loss_)
            label_test = np.array(label_test)
            y_array = np.array([tensor.item() for tensor in y])
            fpr_ab, tpr_ab, _ = roc_curve(y_array, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)   
            # print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
            test_roc_abs.append((test_roc_ab, epoch))
            times.append(time.time() - time_0)

        if (epoch)%25 == 0:
            print(f"Epoch {epoch} complete.")

        if epoch == (args.num_epochs-1):
            auroc_final =  test_roc_ab
            print(auroc_final)
    
    return test_roc_abs, times

if __name__ == '__main__':
    args = arg_parse()
    # DS = args.DS
    setup_seed(0)

    if True:
        datasets = ["Tox21_MMP", "Tox21_HSE", "Tox21_p53", "Tox21_PPAR-gamma"]
        for DS in datasets:
            print(f"Testing dataset: {DS}")
            graphs_train_ = load_data.read_graphfile('dataset', DS+'_training', max_nodes=args.max_nodes)  
            graphs_test = load_data.read_graphfile('dataset', DS+'_testing', max_nodes=args.max_nodes)  
            datanum = len(graphs_train_) + len(graphs_test)    
            
            if args.max_nodes == 0:
                max_nodes_num_train = max([G.number_of_nodes() for G in graphs_train_])
                max_nodes_num_test = max([G.number_of_nodes() for G in graphs_test])
                max_nodes_num = max([max_nodes_num_train, max_nodes_num_test])
            else:
                max_nodes_num = args.max_nodes

            graphs_train = []
            for graph in graphs_train_:
                if graph.graph['label'] == 1:
                    graphs_train.append(graph)
            for graph in graphs_train:
                graph.graph['label'] = 0
                    
            graphs_test_nor = []
            graphs_test_ab = []
            for graph in graphs_test:
                if graph.graph['label'] == 0:
                    graphs_test_nor.append(graph)
                else:
                    graphs_test_ab.append(graph)
            for graph in graphs_test_nor:
                graph.graph['label'] = 0
            for graph in graphs_test_ab:
                graph.graph['label'] = 1
                graphs_test_nor.append(graph)
            graphs_test = graphs_test_nor
                        
            num_train = len(graphs_train)
            num_test = len(graphs_test)
            # print(num_train, num_test)

                
            dataset_sampler_train = GraphSampler(graphs_train, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)

            learning_rates = [0.0001]

            params = [
                [2, [16, 8], [8, 8], [3, 1]],
                [2, [16, 8], [8, 8], [3, 2]],
                [2, [16, 16], [8, 8], [3, 1]],
                [2, [16, 16], [8, 8], [3, 2]],
                [2, [16, 8], [6, 6], [3, 1]],
                [2, [16, 8], [6, 6], [3, 2]],
                [2, [16, 8], [4, 4], [3, 1]],
                [2, [16, 8], [4, 4], [3, 2]],
                [3, [16, 8, 8], [8, 4, 4], [3, 3, 2]],
                [3, [16, 8, 4], [8, 4, 4], [3, 3, 2]],
                [3, [16, 8, 8], [8, 4, 4], [3, 2, 1]],
                [3, [16, 8, 4], [8, 4, 4], [3, 3, 1]],
            ]

            for lr, param in product(learning_rates, params):
                print("Printing parameters...")
                print(param)

                model_teacher = gcn.DiffPool(
                    num_features=dataset_sampler_train.feat_dim, 
                    num_pools=param[0],
                    hidden_dims=param[1],
                    embedding_dims=param[2],
                    num_layers=param[3]
                )
                
                for p in model_teacher.parameters():
                    p.requires_grad = False

                model_student = gcn.DiffPool(
                    num_features=dataset_sampler_train.feat_dim, 
                    num_pools=param[0],
                    hidden_dims=param[1],
                    embedding_dims=param[2],
                    num_layers=param[3]
                )

                    
                data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, 
                                                                shuffle=True,
                                                                batch_size=args.batch_size)

                
                dataset_sampler_test = GraphSampler(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
                data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, 
                                                                    shuffle=False,
                                                                    batch_size=1)
                tests, times = train(data_train_loader, data_test_loader, model_teacher, model_student, args.num_epochs, lr, args)   

                print("Max roc: ", end="")
                print(sorted(tests, key=lambda x: x[0], reverse=True)[0])

    if True:
        datasets = ["Tox21_MMP", "Tox21_HSE", "Tox21_p53", "Tox21_PPAR-gamma"]
        params = {}
        params["Tox21_MMP"] = [2, [16, 8], [8, 8], [3, 2]]
        params["Tox21_HSE"] = [3, [16, 8, 8], [8, 4, 4], [3, 3, 2]]
        params["Tox21_p53"] = [3, [16, 8, 4], [8, 4, 4], [3, 3, 1]]
        params["Tox21_PPAR-gamma"] = [3, [16, 8, 4], [8, 4, 4], [3, 3, 1]]

        for DS in datasets:
            param = params[DS]

            graphs_train_ = load_data.read_graphfile('dataset', DS+'_training', max_nodes=args.max_nodes)  
            graphs_test = load_data.read_graphfile('dataset', DS+'_testing', max_nodes=args.max_nodes)  
            datanum = len(graphs_train_) + len(graphs_test)    
            
            if args.max_nodes == 0:
                max_nodes_num_train = max([G.number_of_nodes() for G in graphs_train_])
                max_nodes_num_test = max([G.number_of_nodes() for G in graphs_test])
                max_nodes_num = max([max_nodes_num_train, max_nodes_num_test])
            else:
                max_nodes_num = args.max_nodes

            graphs_train = []
            for graph in graphs_train_:
                if graph.graph['label'] == 1:
                    graphs_train.append(graph)
            for graph in graphs_train:
                graph.graph['label'] = 0
                    
            graphs_test_nor = []
            graphs_test_ab = []
            for graph in graphs_test:
                if graph.graph['label'] == 0:
                    graphs_test_nor.append(graph)
                else:
                    graphs_test_ab.append(graph)
            for graph in graphs_test_nor:
                graph.graph['label'] = 0
            for graph in graphs_test_ab:
                graph.graph['label'] = 1
                graphs_test_nor.append(graph)
            graphs_test = graphs_test_nor
                        
            num_train = len(graphs_train)
            num_test = len(graphs_test)
            # print(num_train, num_test)

            dataset_sampler_train = GraphSampler(graphs_train, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)

            model_teacher = gcn.DiffPool(
                        num_features=dataset_sampler_train.feat_dim, 
                        num_pools=param[0],
                        hidden_dims=param[1],
                        embedding_dims=param[2],
                        num_layers=param[3]
                    )
                    
            for p in model_teacher.parameters():
                        p.requires_grad = False

            model_student = gcn.DiffPool(
                        num_features=dataset_sampler_train.feat_dim, 
                        num_pools=param[0],
                        hidden_dims=param[1],
                        embedding_dims=param[2],
                        num_layers=param[3]
                    )

                        
            data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, 
            shuffle=True,
            batch_size=args.batch_size)

                    
            dataset_sampler_test = GraphSampler(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
            
            data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, 
                                                            shuffle=False,
                                                            batch_size=1)
            tests_pool, times_pool = train(data_train_loader, data_test_loader, model_teacher, model_student, args.num_epochs, lr, args)

            print("Max roc: ", end="")
            print(sorted(tests_pool, key=lambda x: x[0], reverse=True)[0])


            dataset_sampler_train = GraphSampler(graphs_train, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        
            model_teacher = gcn.GCN(dataset_sampler_train.feat_dim, 16, args.output_dim, 2, 3, bn=args.bn, args=args)
            for param in model_teacher.parameters():
                param.requires_grad = False
        
            model_student = gcn.GCN(dataset_sampler_train.feat_dim, 16, args.output_dim, 2, 3, bn=args.bn, args=args)
                
            data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, 
                                                            shuffle=True,
                                                            batch_size=args.batch_size)

            
            dataset_sampler_test = GraphSampler(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
            
            data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, 
                                                                shuffle=False,
                                                                batch_size=1)

            tests_gcn, times_gcn = train_regular(data_train_loader, data_test_loader, model_teacher, model_student, args.num_epochs, args)
            print("Max roc: ", end="")
            print(sorted(tests_gcn, key=lambda x: x[0], reverse=True)[0])

            f = open("test_results", "a")
            f.write(f"Printing for dataset {DS}\n")
            f.write("Wthout DiffPool\n")
            f.write(str(tests_gcn) + "\n")
            f.write("DiffPooled\n")
            f.write(str(tests_pool) + "\n")
            f.close()

            tests_gcn = [t for t, _ in tests_gcn]
            tests_pool = [t for t, _ in tests_pool]

            plot(tests_gcn, times_gcn, tests_pool, times_pool, param, lr, DS)

                    
                    
