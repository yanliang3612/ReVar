import torch
# from src.argument import parse_args
# args = parse_args()



class Imbalance:
    def __init__(self, name,data, ratio):
        self.name = name
        self.data = data
        self.total_node = len(data.x)
        self.label = data.y
        self.ratio = int(ratio)
        self.data_train_mask = data.train_mask.clone()
        self.n_cls = data.y.max().item() + 1

        self._class_num_list = self.class_num_list()
        self._idx_info = self.get_idx_info()
        self._n_data = self.n_data()

    def n_data(self):
        n_data = []
        stats = self.data.y[self.data_train_mask]
        for i in range(self.n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        return n_data


    def class_num_list(self):
        class_num_list = []
        if self.ratio == 20 or self.ratio == 10:
            if self.name == 'Cora':
                class_sample_num = 20
                imb_class_num = 3
            elif self.name == 'CiteSeer':
                class_sample_num = 20
                imb_class_num = 3
            elif self.name == 'PubMed':
                class_sample_num = 20
                imb_class_num = 1
            elif self.name == 'Computers':
                class_sample_num = 20
                imb_class_num = 5
            elif self.name == 'Photo':
                class_sample_num = 20
                imb_class_num = 4
            elif self.name == 'CS':
                class_sample_num = 20
                imb_class_num = 7
            else:
                print("no this dataset: {args.dataset}")

        if  self.ratio == 50 :
            if self.name == 'Cora':
                class_sample_num = 50
                imb_class_num = 3
            elif self.name == 'CiteSeer':
                class_sample_num = 50
                imb_class_num = 3
            elif self.name == 'PubMed':
                class_sample_num = 50
                imb_class_num = 1
            elif self.name == 'Computers':
                class_sample_num = 50
                imb_class_num = 5
            elif self.name == 'Photo':
                class_sample_num = 50
                imb_class_num = 4
            elif self.name == 'CS':
                class_sample_num = 50
                imb_class_num = 7
            else:
                print("no this dataset: {args.dataset}")


        if  self.ratio == 100 :
            if self.name == 'Cora':
                class_sample_num = 100
                imb_class_num = 3
            elif self.name == 'CiteSeer':
                class_sample_num = 100
                imb_class_num = 3
            elif self.name == 'PubMed':
                class_sample_num = 100
                imb_class_num = 1
            elif self.name == 'Computers':
                class_sample_num = 100
                imb_class_num = 5
            elif self.name == 'Photo':
                class_sample_num = 100
                imb_class_num = 4
            elif self.name == 'CS':
                class_sample_num = 100
                imb_class_num = 7
            else:
                print("no this dataset: {args.dataset}")

        for i in range(self.n_cls):
            if self.ratio > 1 and i > self.n_cls - 1 - imb_class_num:  # only imbalance the last classes
                class_num_list.append(int(class_sample_num * (1. / self.ratio)))
            else:
                class_num_list.append(class_sample_num)

        return class_num_list

    def get_idx_info(self):
        index_list = torch.arange(len(self.label))
        idx_info = []
        for i in range(self.n_cls):
            cls_indices = index_list[((self.label == i) & self.data_train_mask)]
            idx_info.append(cls_indices)
        return idx_info

    def split_semi_dataset(self):
        n_data = self._n_data
        class_num_list = self._class_num_list
        idx_info = self._idx_info

        new_idx_info = []
        _train_mask = idx_info[0].new_zeros(self.total_node, dtype=torch.bool)
        for i in range(self.n_cls):
            if n_data[i] > class_num_list[i]:
                cls_idx = torch.randperm(len(idx_info[i]))
                cls_idx = idx_info[i][cls_idx]
                cls_idx = cls_idx[:class_num_list[i]]
                new_idx_info.append(cls_idx)
            else:
                new_idx_info.append(idx_info[i])
            _train_mask[new_idx_info[i]] = True
        assert _train_mask.sum().long() == sum(class_num_list)
        assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

        return _train_mask





class Imbalance_(Imbalance):
    '''
    It has basically the same function as Imbalance, 
    but it will add attributes to `data` when the method is called.
    Therefore, it can produce data for further usage like feature-dropping to generate a balanced graph.
    '''
    def __init__(self, name,data, ratio):
        super().__init__(name, data, ratio)

    def add_attributes_to_data(self):
        # add attributes to data
        self.data.ratio = self.ratio # imbalance ratio
        self.data.n_cls = self.n_cls # class number
        self.data.class_num_list = self.class_num_list() # node number of each class
        self.data.idx_info = self.get_idx_info() # a list containing node indices of each class

    def n_data(self):
        n_data = super().n_data()
        self.data.n_data = n_data
        return n_data

    def class_num_list(self):
        class_num_list = super().class_num_list()
        self.data.class_num_list = class_num_list
        return class_num_list

    def get_idx_info(self):
        idx_info = super().get_idx_info()
        self.data.idx_info = idx_info
        return idx_info

    def split_semi_dataset(self):
        n_data = self.n_data()
        class_num_list = self.class_num_list()
        idx_info = self.get_idx_info()
        new_idx_info = []
        _train_mask = idx_info[0].new_zeros(self.total_node, dtype=torch.bool)
        for i in range(self.n_cls):
            if n_data[i] > class_num_list[i]:
                cls_idx = torch.randperm(len(idx_info[i]))
                cls_idx = idx_info[i][cls_idx]
                cls_idx = cls_idx[:class_num_list[i]]
                new_idx_info.append(cls_idx)
            else:
                new_idx_info.append(idx_info[i])
            _train_mask[new_idx_info[i]] = True
        assert _train_mask.sum().long() == sum(class_num_list)
        assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

        self.data.imbalanced_idx_info = new_idx_info

        return _train_mask

    def get_balanced_mask(self):
        n_data = self._n_data
        class_num_list = self._class_num_list
        num_per_class = min(class_num_list)
        idx_info = self.data.imbalanced_idx_info
        
        balanced_mask = idx_info[0].new_ones(self.total_node, dtype=torch.bool)
        for i in range(self.n_cls):
            if n_data[i] > num_per_class:
                discard_cls_idx = torch.randperm(len(idx_info[i]))
                discard_cls_idx = idx_info[i][discard_cls_idx]
                discard_cls_idx = discard_cls_idx[num_per_class:]
                balanced_mask[discard_cls_idx] = False

        return balanced_mask