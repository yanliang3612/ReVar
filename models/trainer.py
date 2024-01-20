from .RVGNN import RVGNN
import torch
from src.transform import get_graph_drop_transform
from src.utils import compute_accuracy
import os.path as osp
import statistics

import torch.nn.functional as F
from torch.optim import Adam
from src.sampling import Sampler
from src.utils import set_random_seeds
from copy import deepcopy
from src.data import Planetoid
from src.imbalance import Imbalance, Imbalance_
from src.data import data_mask,data_mask_computersrandom,data_mask_csrandom
import torch_geometric.transforms as T
from layers import GNN, Classifier
from src.classcenter import classcenter
from src.supconloss import supconloss2
from src.data import make_longtailed_data_remove,more_information



class Base_Trainer:
    def __init__(self, args):

        self.args = args
        self.device =torch.device(f'cuda:{self.args.device}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)

        self.path = osp.join(args.datadir, self.args.dataset)
 
        self.hidden_layers = [self.args.dim for i in range(self.args.layers)]


        # For Evaluation
        self.best_val = 0
        self.epoch_list = []  # for epoch select

        self.train_accs = []
        self.valid_accs = []
        self.test_accs = []


        self.train_baccs = []
        self.valid_baccs = []
        self.test_baccs = []


        self.train_f1 = []
        self.valid_f1 = []
        self.test_f1 = []


        self.running_train_accs = []
        self.running_valid_accs = []
        self.running_test_accs = []

        self.running_train_baccs = []
        self.running_valid_baccs = []
        self.running_test_baccs = []

        self.running_train_f1 = []
        self.running_valid_f1 = []
        self.running_test_f1 = []

        
    def evaluate(self, batch_data, st, epoch=-1):

        # Classifier Accuracy
        self.model.eval()
        _, preds = self.model.cls(batch_data)   # 分类结果

        train_acc, val_acc, test_acc,train_bacc,val_bacc,test_bacc,train_f1,val_f1,test_f1 = compute_accuracy(preds, batch_data.y, self.train_mask, self.val_mask,
                                                        self.test_mask)      #
        self.running_train_accs.append(train_acc)                            #
        self.running_valid_accs.append(val_acc)
        self.running_test_accs.append(test_acc)

        self.running_train_baccs.append(train_bacc)
        self.running_valid_baccs.append(val_bacc)
        self.running_test_baccs.append(test_bacc)

        self.running_train_f1.append(train_f1)
        self.running_valid_f1.append(val_f1)
        self.running_test_f1.append(test_f1)


        if val_acc > self.best_val:
            self.best_val = val_acc
            self.cnt = 0
        else:
            self.cnt += 1

        st += '| train_acc: {:.2f} | valid_acc : {:.2f} | test_acc : {:.2f}| ' \
              'train_bacc: {:.2f} | valid_bacc : {:.2f} | test_bacc : {:.2f}' \
              '| train_f1: {:.2f} | valid_f1 : {:.2f} | test_f1 : {:.2f}' \
            .format(train_acc, val_acc, test_acc,train_bacc,val_bacc,test_bacc,train_f1,val_f1,test_f1)
        if epoch % 50 == 0:
            print(st)


    def save_results(self, fold):

        train_acc, val_acc, test_acc= torch.tensor(self.running_train_accs), torch.tensor(
            self.running_valid_accs), torch.tensor(self.running_test_accs)

        train_bacc, val_bacc, test_bacc = torch.tensor(self.running_train_baccs), torch.tensor(
            self.running_valid_baccs), torch.tensor(self.running_test_baccs)

        train_f1, val_f1, test_f1 = torch.tensor(self.running_train_f1), torch.tensor(
            self.running_valid_f1), torch.tensor(self.running_test_f1)

        selected_epoch = val_acc.argmax()

        best_train_acc = train_acc[selected_epoch]
        best_val_acc = val_acc[selected_epoch]
        best_test_acc = test_acc[selected_epoch]

        best_train_bacc = train_bacc[selected_epoch]
        best_val_bacc = val_bacc[selected_epoch]
        best_test_bacc = test_bacc[selected_epoch]

        best_train_f1 = train_f1[selected_epoch]
        best_val_f1 = val_f1[selected_epoch]
        best_test_f1 = test_f1[selected_epoch]




        self.epoch_list.append(selected_epoch.item())
        self.train_accs.append(best_train_acc.item())
        self.valid_accs.append(best_val_acc.item())
        self.test_accs.append(best_test_acc.item())

        self.train_baccs.append(best_train_bacc.item())
        self.valid_baccs.append(best_val_bacc.item())
        self.test_baccs.append(best_test_bacc.item())

        self.train_f1.append(best_train_f1.item())
        self.valid_f1.append(best_val_f1.item())
        self.test_f1.append(best_test_f1.item())


        if fold + 1 != self.args.repetitions:
            self.running_train_accs = []
            self.running_valid_accs = []
            self.running_test_accs = []

            self.running_train_baccs = []
            self.running_valid_baccs = []
            self.running_test_baccs = []

            self.running_train_f1 = []
            self.running_valid_f1 = []
            self.running_test_f1 = []

            self.cnt = 0
            self.best_val = 0


    def summary(self):
        if len(self.train_accs) == 1:
            train_acc_mean = self.train_accs[0]
            val_acc_mean = self.valid_accs[0]
            test_acc_mean = self.test_accs[0]

            val_f1_mean = self.valid_f1[0]
            test_f1_mean = self.test_f1[0]
            test_bacc_mean = self.test_baccs[0]

            acc_CI = 0
            bacc_CI = 0
            f1_CI = 0

        else:
            train_acc_mean = statistics.mean(self.train_accs)
            val_acc_mean = statistics.mean(self.valid_accs)
            test_acc_mean = statistics.mean(self.test_accs)

            val_f1_mean = statistics.mean(self.valid_f1)
            test_f1_mean = statistics.mean(self.test_f1)
            test_bacc_mean =statistics.mean(self.test_baccs)

            acc_CI = (statistics.stdev(self.test_accs) / (self.args.repetitions ** (1 / 2)))
            bacc_CI = (statistics.stdev(self.test_baccs) / (self.args.repetitions ** (1 / 2)))
            f1_CI = (statistics.stdev(self.test_f1) / (self.args.repetitions ** (1 / 2)))

        log= "** | test acc : {:.2f} +- {:.2f} | test bacc : {:.2f} +- {:.2f} | test f1 : {:.2f} +- {:.2f} |val acc: {:.2f} |val f1: {:.2f} |train acc: {:.2f} | **\n".format(
            test_acc_mean, acc_CI, test_bacc_mean,bacc_CI,test_f1_mean,f1_CI,val_acc_mean,val_f1_mean, train_acc_mean)
        log_dict = {"test_acc_mean": test_acc_mean,
                    "test_acc_std": acc_CI,
                    "test_bacc_mean": test_bacc_mean,
                    "test_bacc_std": bacc_CI,
                    "test_f1_mean": test_f1_mean,
                    "test_f1_std": f1_CI,
                    "val_acc": val_acc_mean,
                    "val_f1": val_f1_mean,
                    "train_acc": train_acc_mean,
                    "log_txt": log}
        return log_dict



















class RVGNN_Trainer(Base_Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.transform1 = get_graph_drop_transform(drop_edge_p=args.de_1, drop_feat_p=args.df_1)
        self.transform2 = get_graph_drop_transform(drop_edge_p=args.de_2, drop_feat_p=args.df_2)

        self.args = args

    def _init_model(self):

        self.model = RVGNN(self.encoder, self.classifier, self.unique_labels, self.args.tau, self.args.thres, device=self.args.device).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay)


    def _init_dataset(self):
        if self.args.dataset == 'Cora' or self.args.dataset == 'CiteSeer' or self.args.dataset == 'PubMed':
            self.data = \
                Planetoid(self.path, self.args.dataset, transform=T.NormalizeFeatures(), split='public',
                          )[0].to(
                    self.device)
            self.data.imb_ratio = self.args.imb_ratio
            self.imbalance_handler =  Imbalance_(self.args.dataset, self.data, self.args.imb_ratio)
            self.data.train_mask =self.imbalance_handler.split_semi_dataset()
            self.data.train_mask = self.data.train_mask.to(self.device)
            self.n_data, self.n_cls = more_information(self.data)
            self.class_num_list, self.data.train_mask, self.idx_info, self.train_node_mask, self.train_edge_mask = make_longtailed_data_remove(self.data.edge_index,self.data.y,self.n_data,self.n_cls,self.args.imb_ratio,self.data.train_mask)
            self.data.edge_index = self.data.edge_index[:,self.train_edge_mask]


        elif self.args.dataset == 'Computers-semi' :
            from torch_geometric.datasets import Amazon
            self.data = Amazon(self.path, self.args.dataset)[0]
            self.data.imb_ratio = self.args.imb_ratio
            self.data.train_mask, self.data.val_mask, self.data.test_mask = data_mask(self.data)
            self.data.train_mask, self.data.val_mask, self.data.test_mask = torch.squeeze(self.data.train_mask, 1), \
                                                             torch.squeeze(self.data.val_mask, 1), \
                                                             torch.squeeze(self.data.test_mask, 1)
            self.data.train_mask = Imbalance(self.args.dataset, self.data, self.args.imb_ratio).split_semi_dataset()
            self.data = self.data.to(self.device)
            self.data.train_mask, self.data.val_mask, self.data.test_mask = self.data.train_mask.to(self.device), \
                                                                            self.data.val_mask.to(self.device), \
                                                                           self.data.test_mask.to(self.device)
        elif self.args.dataset == 'Computers-random' :
            from torch_geometric.datasets import Amazon
            self.data = Amazon(self.path, self.args.dataset)[0]
            self.data.train_mask, self.data.val_mask, self.data.test_mask = data_mask_computersrandom(self.data)
            self.data.train_mask, self.data.val_mask, self.data.test_mask = torch.squeeze(self.data.train_mask, 1), \
                                                             torch.squeeze(self.data.val_mask, 1), \
                                                             torch.squeeze(self.data.test_mask, 1)
            self.data = self.data.to(self.device)
            self.data.train_mask, self.data.val_mask, self.data.test_mask = self.data.train_mask.to(self.device), \
                                                                            self.data.val_mask.to(self.device), \
                                                                           self.data.test_mask.to(self.device)

        elif self.args.dataset == 'CS-random' :
            from torch_geometric.datasets import Coauthor
            self.data = Coauthor(self.path, self.args.dataset)[0]
            self.data.train_mask, self.data.val_mask, self.data.test_mask = data_mask_csrandom(self.data)
            self.data.train_mask, self.data.val_mask, self.data.test_mask = torch.squeeze(self.data.train_mask, 1), \
                                                             torch.squeeze(self.data.val_mask, 1), \
                                                             torch.squeeze(self.data.test_mask, 1)
            self.data = self.data.to(self.device)
            self.data.train_mask, self.data.val_mask, self.data.test_mask = self.data.train_mask.to(self.device), \
                                                                            self.data.val_mask.to(self.device), \
                                                                            self.data.test_mask.to(self.device)


    def train(self):
        
        for fold in range(self.args.repetitions):

            set_random_seeds(fold)

            self._init_dataset()

            self.Sampler = Sampler(self.args, self.data, self.labels, self.running_train_mask)

            input_size = self.data.x.size(1)

            rep_size = self.hidden_layers[-1]

            self.unique_labels = self.data.y.unique()

            self.num_classes = len(self.unique_labels)

            self.encoder = GNN([input_size] + self.hidden_layers, args=self.args)

            self.classifier = Classifier(rep_size, self.num_classes)

            self._init_model()

            for epoch in range(1, self.args.epochs+1):


                self.model.train()

                self.optimizer.zero_grad()

                anchor = self.transform1(self.data)
                positive = self.transform2(self.data)


                label_matrix, support_index, self.batch_size = self.Sampler.sample()

                anchor_rep = self.model(anchor)


                pos_rep = self.model(positive)


                anchor_support_rep = anchor_rep[support_index]


                pos_support_rep = pos_rep[support_index]

                anchor_centroid, pos_centroid, matrix = classcenter(self.num_classes, anchor_rep, pos_rep, self.labels,self.train_mask, self.device)

                if self.args.balancedmask:

                    snn_mask = 0

                else:

                    snn_mask = self.running_train_mask

                if self.args.classcenter:

                    consistency_loss = self.model.loss(anchor_rep, pos_rep, anchor_centroid, pos_centroid, matrix, self.data.y, snn_mask)

                else:

                    consistency_loss = self.model.loss(anchor_rep, pos_rep, anchor_support_rep, pos_support_rep,label_matrix, self.data.y, snn_mask)


                sup_loss = 0.

                logits, _ = self.model.cls(anchor)

                sup_loss += F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                logits, _ = self.model.cls(positive)
                sup_loss += F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                sup_loss /= 2
                # unsupervised loss
                if self.args.supervised:
                    # unsup_loss = supconloss(anchor_rep, pos_rep,self.running_train_mask,self.labels,self.num_classes)
                    unsup_loss = supconloss2(anchor_rep, pos_rep, self.running_train_mask, self.labels, self.num_classes)
                else:

                    unsup_loss = 2 - 2* F.cosine_similarity(anchor_rep, pos_rep, dim=-1).mean()
                loss = sup_loss + self.args.lam*consistency_loss + self.args.lam2 * unsup_loss
                loss.backward()
                self.optimizer.step()

                
                #
                st = '[Fold : {}][Epoch {}/{}] Consistency_Loss: {:.4f} | Sup_loss : {:.4f} | Unsup_loss : {:.4f} | Total_loss : {:.4f}'.format(
                        fold+1, epoch, self.args.epochs, consistency_loss.item(), sup_loss.item(), unsup_loss.item(), loss.item())
                
                # evaluation
                self.evaluate(self.data, st, epoch)
                #
                if self.cnt == self.args.patience:
                    # print("early stopping!")
                    break

            self.save_results(fold)
            
        return self.summary()

