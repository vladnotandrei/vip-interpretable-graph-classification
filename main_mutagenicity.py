import argparse
import random
# import time
# import glob
from tqdm import tqdm   
import os

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from arch.mutagenicity import ClassifierMutagenicity, QuerierMutagenicity
import dataset
import ops
import utils
import wandb


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data', type=str, default='mutagenicity')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_queries', type=int, default=100)
    parser.add_argument('--max_queries_test', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='mutagenicity')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/Mutagenicity/', help='save directory')
    parser.add_argument('--query_dir', type=str, default='./experiments/rdkit_querysets/queryset_1.csv', help='save directory')
    parser.add_argument('--ckpt_dir', type=bool, default=None, help='load checkpoint from this dir') 
    args = parser.parse_args()
    return args


def adaptive_sampling(x, max_queries, model):
    model.requires_grad_(False)  # work around for unused parameter error
    device = x.device
    N, D = x.shape  # (bs, n_queries)
    
    rand_history_length = torch.randint(low=0, high=max_queries, size=(N, )).to(device)  # (bs)
    mask = torch.zeros((N, D), requires_grad=False).to(device)  # (bs, n_queries)
    for _ in range(max_queries): # +1 because we start from empty history (What does this comment mean ??)
        masked_input = x * mask  # (bs, n_queries)
        with torch.no_grad():
            query = model(masked_input, mask)  # (bs, n_queries) onehot tensors
                
        # index only the rows smaller than rand_history_length
        idx = mask.sum(1) <= rand_history_length  # (bs) boolean tensor
        mask[idx] = mask[idx] + query[idx]
    model.requires_grad_(True)  # work around for unused parameter error
    return mask  # (bs, n_queries) tensor where there are max_queries elements in dim=1 that are > 0


def evaluate(test_features, querier, classifier, n_queries, max_queries_test):
    """
    Input:
    test_features: (batch_size, n_queries)

    Output:
    logits: (batch_size, max_queries_test, 2)
    queries: (batch_size, max_queries_test, n_queries)
    """
    querier.eval()
    classifier.eval()

    # Compute logits for all queries
    test_bs = test_features.shape[0]
    mask = torch.zeros(test_bs, n_queries).to(test_features.device)
    logits, queries = [], []
    for i in range(max_queries_test):
        with torch.no_grad():
            query = querier(test_features * mask, mask)
            label_logits = classifier(test_features * (mask + query))

        mask[np.arange(test_bs), query.argmax(dim=1)] = 1.0
        
        logits.append(label_logits)
        queries.append(query)   
    logits = torch.stack(logits).permute(1, 0, 2)  # (batch_size, max_queries_test, 2)
    queries = torch.stack(queries).permute(1, 0, 2)  # (batch_size, max_queries_test, n_queries)
    return logits, queries


def main(args):
    ## Setup
    wandb
    run = wandb.init(project="Variational-IP", name=args.name, mode=args.mode)
    model_dir = os.path.join(args.save_dir, f'{run.id}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    wandb.config.update(args)

    # cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)

    # random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## constants
    N_QUERIES = 403  # Would be nice to make a param if we change query set
    THRESHOLD = args.threshold

    ## Data
    trainset, testset = dataset.load_mutagenicity_queryset(args.data_dir, args.query_dir, train_ratio=0.8, seed=args.seed)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)

    ## Model
    classifier = ClassifierMutagenicity(queryset_size=N_QUERIES)
    classifier = nn.DataParallel(classifier).to(device)  # What is this?
    querier = QuerierMutagenicity(queryset_size=N_QUERIES, tau=args.tau_start)
    querier = nn.DataParallel(querier).to(device)

    ## Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(querier.parameters()) + list(classifier.parameters()), 
                           amsgrad=True, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    ## Load checkpoint
    if args.ckpt_path is not None:
        ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
        classifier.load_state_dict(ckpt_dict['classifier'])
        querier.load_state_dict(ckpt_dict['querier'])
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        scheduler.load_state_dict(ckpt_dict['scheduler'])
        print('Checkpoint Loaded!')

    ## Train
    for epoch in range(args.epochs):

        # training
        classifier.train()
        querier.train()
        tau = tau_vals[epoch]
        for train_features, train_labels, _ in tqdm(trainloader):  # 3rd elem is dataset id for ref to raw data
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)
            train_bs = train_features.shape[0]
            querier.module.update_tau(tau)
            optimizer.zero_grad()

            # initial random sampling
            if args.sampling == 'biased':
                # TODO: Do I need to pass seed here?
                mask = adaptive_sampling(train_features, args.max_queries, querier).to(device).float()
                # TODO: DO I need to pass seed here?
            elif args.sampling == 'random':
                mask = ops.random_sampling(args.max_queries, N_QUERIES, train_bs).to(device).float()
            history = train_features * mask
            
            # Query and update
            query = querier(history, mask)
            updated_history = history + (train_features * query)

            # prediction
            train_logits = classifier(updated_history)

            # backprop
            loss = criterion(train_logits, train_labels)
            loss.backward()
            optimizer.step()

            # logging
            wandb.log({
                'epoch': epoch,
                'loss': loss.item(),
                'lr': utils.get_lr(optimizer),
                'gradnorm_cls': utils.get_grad_norm(classifier),
                'gradnorm_qry': utils.get_grad_norm(querier)
                })
        scheduler.step()

        # saving
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'classifier': classifier.state_dict(),
                'querier': querier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                },
                os.path.join(model_dir, 'ckpt', f'epoch{epoch}.ckpt'))

        # evaluation
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # classifier.eval()
            # querier.eval()
            epoch_test_qry_need = []
            epoch_test_acc_max = 0
            epoch_test_acc_ip = 0
            for test_features, test_labels, _ in tqdm(testloader):
                test_features = test_features.to(device)
                test_labels = test_labels.to(device)

                params = {
                    'test_features': test_features,
                    'querier': querier,
                    'classifier': classifier,
                    'n_queries': N_QUERIES,
                    'max_queries_test': args.max_queries_test,
                }
                logits, queries = evaluate(**params)
                
                # test_features = test_features.to(device)
                # test_labels = test_labels.to(device)
                # test_bs = test_features.shape[0]

                # # Compute logits for all queries
                # mask = torch.zeros(test_bs, N_QUERIES).to(device)
                # logits, queries = [], []
                # for i in range(args.max_queries_test):
                #     with torch.no_grad():
                #         query = querier(test_features * mask, mask)
                #         label_logits = classifier(test_features * (mask + query))

                #     mask[np.arange(test_bs), query.argmax(dim=1)] = 1.0
                    
                #     logits.append(label_logits)
                #     queries.append(query)   
                # logits = torch.stack(logits).permute(1, 0, 2)

                # accuracy using all queries
                test_pred_max = logits[:, -1, :].argmax(dim=1).float()
                test_acc_max = (test_pred_max == test_labels.squeeze()).float().sum()
                epoch_test_acc_max += test_acc_max

                # compute number of queries needed for prediction
                qry_need = ops.compute_queries_needed(logits, threshold=THRESHOLD)
                epoch_test_qry_need.append(qry_need)

                # accuracy using IP
                test_pred_ip = logits[torch.arange(len(qry_need)), qry_need-1].argmax(1)
                test_acc_ip = (test_pred_ip == test_labels.squeeze()).float().sum()
                epoch_test_acc_ip += test_acc_ip
            epoch_test_acc_max = epoch_test_acc_max / len(testset)
            epoch_test_acc_ip = epoch_test_acc_ip / len(testset)

            # mean and std of queries needed
            epoch_test_qry_need = torch.hstack(epoch_test_qry_need).float()
            qry_need_avg = epoch_test_qry_need.mean()
            qry_need_std = epoch_test_qry_need.std()

            # logging
            wandb.log({
                'test_epoch': epoch,
                'test_acc_max': epoch_test_acc_max,
                'test_acc_ip': epoch_test_acc_ip,
                'qry_need_avg': qry_need_avg,
                'qry_need_std': qry_need_std
            })


if __name__ == '__main__':
    args = parseargs()    
    main(args)