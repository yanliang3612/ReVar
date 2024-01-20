import os
import copy
import torch
from src.utils import config2string
from src.transform import get_graph_drop_transform
from src.utils import compute_accuracy
from layers import GNN, Classifier
import os.path as osp
import statistics


class embedder:
    def __init__(self, args):

        self.args = args
        self.device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))

        self.path = osp.join('/tmp', 'data', self.args.dataset)

        self.transform1 = get_graph_drop_transform(drop_edge_p=args.de_1, drop_feat_p=args.df_1)
        self.transform2 = get_graph_drop_transform(drop_edge_p=args.de_2, drop_feat_p=args.df_2)

        if  self.args.layers == 1:
            self.hidden_layers = [self.args.dim]
        elif self.args.layers == 2:
            self.hidden_layers=[self.args.dim,self.args.dim]
        elif self.args.layers == 3:
            self.hidden_layers = [self.args.dim, self.args.dim,self.args.dim]


        # For Evaluation
        self.best_val = 0
        self.epoch_list = []  # for epoch select

        self.train_accs = [];
        self.valid_accs = [];
        self.test_accs = []


        self.train_baccs = [];
        self.valid_baccs = [];
        self.test_baccs = []


        self.train_f1 = [];
        self.valid_f1 = [];
        self.test_f1 = []


        self.running_train_accs = [];
        self.running_valid_accs = [];
        self.running_test_accs = []

        self.running_train_baccs = [];
        self.running_valid_baccs = [];
        self.running_test_baccs = []

        self.running_train_f1 = [];
        self.running_valid_f1 = [];
        self.running_test_f1 = [];

        
    def evaluate(self, batch_data, st):

        # Classifier Accuracy
        # Classifier Accuracy
        self.model.eval()
        _, preds = self.model.cls(batch_data)

        train_acc, val_acc, test_acc,train_bacc,val_bacc,test_bacc,train_f1,val_f1,test_f1 = compute_accuracy(preds, batch_data.y, self.train_mask, self.val_mask,
                                                        self.test_mask)
        self.running_train_accs.append(train_acc);
        self.running_valid_accs.append(val_acc);
        self.running_test_accs.append(test_acc)

        self.running_train_baccs.append(train_bacc);
        self.running_valid_baccs.append(val_bacc);
        self.running_test_baccs.append(test_bacc)

        self.running_train_f1.append(train_f1);
        self.running_valid_f1.append(val_f1);
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
        self.train_accs.append(best_train_acc.item());
        self.valid_accs.append(best_val_acc.item());
        self.test_accs.append(best_test_acc.item());

        self.train_baccs.append(best_train_bacc.item());
        self.valid_baccs.append(best_val_bacc.item());
        self.test_baccs.append(best_test_bacc.item());

        self.train_f1.append(best_train_f1.item());
        self.valid_f1.append(best_val_f1.item());
        self.test_f1.append(best_test_f1.item());


        if fold + 1 != self.args.repetitions:
            self.running_train_accs = [];
            self.running_valid_accs = [];
            self.running_test_accs = []

            self.running_train_baccs = [];
            self.running_valid_baccs = [];
            self.running_test_baccs = []

            self.running_train_f1 = [];
            self.running_valid_f1 = [];
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
        print(log)
