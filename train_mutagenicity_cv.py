import argparse
import random
from tqdm import tqdm   
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
from arch.mutagenicity import ClassifierMutagenicity, QuerierMutagenicity
from train_mutagenicity import train_step, evaluate
import dataset
import ops
import utils
import wandb
from sklearn.model_selection import KFold

# import train_mutagenicity


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--queryset_size', type=int, help='Number of queries in the query set')
    parser.add_argument('--max_queries', type=int, default=100)
    parser.add_argument('--max_queries_test', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='mutagenicity')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/Mutagenicity/', help='save directory')
    parser.add_argument('--query_dir', type=str, default='./data/rdkit_queryset.csv', help='save directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--ckpt_fold_idx', type=int, default=-1, help='fold idx from 0,...,9 of loaded ckpt')  # if -1, no ckpt_fold_idx
    parser.add_argument('--ckpt_epoch', type=int, default=1, help='The epoch of the ckpt being loaded')
    parser.add_argument('--ckpt_sampling_method', type=str, default=None, help='The sampling method of the ckpt being loaded')
    args = parser.parse_args()
    return args


def main(args):
    # cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)

    # random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## constants
    N_QUERIES = args.queryset_size
    THRESHOLD = args.threshold

    ## 10-fold Cross Validation
    whole_dataset = dataset.load_mutagenicity_query_answer_dataset(args.data_dir, args.query_dir, device, train_ratio=None, seed=None)  # load entire dataset
    dataset_idx_list = range(len(whole_dataset))  # [0,1,2,...,|dataset|-1]
    kfold = KFold(n_splits=10, shuffle=False)

    for fold_idx, (train_index, test_index) in enumerate(kfold.split(dataset_idx_list)):
        sampling_method_options = ['random', 'biased']

        # Skip if already fully trained given fold_idx
        if args.ckpt_path is not None:
            if fold_idx < args.ckpt_fold_idx:
                continue
            elif (args.ckpt_fold_idx == fold_idx):
                if args.ckpt_sampling_method == 'biased':
                    sampling_method_options = ['biased']
                elif args.ckpt_sampling_method == 'random':
                    sampling_method_options = ['random', 'biased']
                else:
                    raise Exception('Invalid ckpt sampling method!') 

        for i, sampling_method in enumerate(sampling_method_options):

            ## Modify epochs left to train if on fold we're loading ckpt for
            epochs = args.epochs  # if not on ckpt fold
            if args.ckpt_path is not None:
                if (args.ckpt_fold_idx == fold_idx) and (args.ckpt_sampling_method == sampling_method):  # On current fold and sampling method we want to load ckpt from
                    epochs = args.epochs - args.ckpt_epoch - 1  # -1 because argsckpt_epoch is 0-indexed, args.epochs is 1-indexed

            ## Setup
            # Start a new wandb run for this fold and sampling method
            run = wandb.init(
                project="Variational-IP",
                name=f'{args.name}_fold={fold_idx}_{sampling_method}',
                mode=args.mode,  # e.g., "online", "offline", or "disabled"
                reinit=True  # ensures wandb doesn't reuse the same run
            )
            model_dir = os.path.join(args.save_dir, f'fold_idx={fold_idx}', f'{sampling_method}', f'{run.id}')
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
            utils.save_params(model_dir, vars(args))
            wandb.config.update(args)
            
            # Set seed for shuffling dataloaders
            dataloader_seed = args.seed + (fold_idx * 10) + i  # unique seed for ech fold and sampling strategy
            g = torch.Generator()
            g.manual_seed(dataloader_seed)

            ## Data
            trainset = Subset(whole_dataset, train_index)
            testset = Subset(whole_dataset, test_index)
            trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, shuffle=True, generator=g)
            testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)

            ## Model
            classifier = ClassifierMutagenicity(queryset_size=N_QUERIES).to(device)
            querier = QuerierMutagenicity(queryset_size=N_QUERIES, tau=args.tau_start).to(device)

            ## Optimization
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(list(querier.parameters()) + list(classifier.parameters()), 
                                amsgrad=True, lr=args.lr)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            tau_vals = np.linspace(args.tau_start, args.tau_end, epochs)

            ## Load checkpoint
            if (args.ckpt_path is not None) and (args.ckpt_fold_idx == fold_idx) and (args.ckpt_sampling_method == sampling_method):  # On current fold we want to load ckpt from
                ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
                classifier.load_state_dict(ckpt_dict['classifier'])
                querier.load_state_dict(ckpt_dict['querier'])
                optimizer.load_state_dict(ckpt_dict['optimizer'])
                scheduler.load_state_dict(ckpt_dict['scheduler'])
                print('Checkpoint Loaded!')
            elif sampling_method == 'biased':
                biased_starting_point = f'{random_model_dir}/ckpt/epoch{args.epochs-1}.ckpt'
                ckpt_dict = torch.load(biased_starting_point, map_location='cpu')
                classifier.load_state_dict(ckpt_dict['classifier'])
                querier.load_state_dict(ckpt_dict['querier'])
                optimizer.load_state_dict(ckpt_dict['optimizer'])
                scheduler.load_state_dict(ckpt_dict['scheduler'])
                print('Biased model starting point loaded!')
            
            # Save random model dir to give to next biased model training
            if sampling_method == 'random':
                random_model_dir = model_dir
                print('Training random sampling model!')

            ## Train
            for epoch in range(epochs):

                # training
                tau = tau_vals[epoch]
                for train_features, train_labels, _ in tqdm(trainloader):  # 3rd elem is dataset id for ref to raw data
                    train_features = train_features.to(device)
                    train_labels = train_labels.to(device)
                    train_bs = train_features.shape[0]

                    params = {
                        'train_features': train_features,
                        'train_labels': train_labels,
                        'train_bs': train_bs,
                        'criterion': criterion,
                        'optimizer': optimizer,
                        'querier': querier,
                        'classifier': classifier,
                        'sampling': sampling_method,
                        'tau': tau,
                        'n_queries': N_QUERIES,
                        'max_queries': args.max_queries,
                        'device': device
                    }
                    loss = train_step(**params)

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
                if epoch % 10 == 0 or epoch == epochs - 1:
                    torch.save({
                        'classifier': classifier.state_dict(),
                        'querier': querier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        },
                        os.path.join(model_dir, 'ckpt', f'epoch{epoch}.ckpt'))

                # evaluation
                if epoch % 10 == 0 or epoch == epochs - 1:
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