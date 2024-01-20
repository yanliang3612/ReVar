import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="/tmp/data")
    parser.add_argument("--dataset", type=str, default="CiteSeer", help="Cora, CiteSeer, PubMed, Computers, Photo,CS")
    
    # masking
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument('--imb_ratio', type=float, default=100, help='Imbalance Ratio')

    # Encoder
    parser.add_argument("--dim", type=int, default=512, help="64,128,256")
    parser.add_argument("--layers", type=int, default=4, help="1,2,3")
    parser.add_argument('--n_head', type=int, default=8,help='the number of heads in GAT')
    parser.add_argument('--net', type=str, default='GCN',help='GCN,GAT,SAGE,SGC,PPNP,CHEB')
    parser.add_argument('--chebgcn_para', type=int, default=2, help=' Chebyshev filter size of ChebConv')


    # optimization
    parser.add_argument("--epochs", '-e', type=int, default=2000, help="The number of epochs")
    parser.add_argument("--lr", '-lr', type=float, default=0.005, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--decay", type=float, default=1e-2, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--patience", type=int, default=200)
    
    # hyper-parameter
    parser.add_argument("--tau", type=float, default=0.13)
    parser.add_argument("--thres", type=float, default=0.6)
    parser.add_argument("--lam", type=float, default=0.25)
    parser.add_argument("--lam2", type=float, default=2.85)

    # augmentation
    parser.add_argument("--df_1", type=float, default=0.4)
    parser.add_argument("--df_2", type=float, default=0.2)
    parser.add_argument("--de_1", type=float, default=0.65)
    parser.add_argument("--de_2", type=float, default=0.35)
    parser.add_argument("--device", type=int, default=0, help="GPU to use")

    # project(for logging)
    parser.add_argument("--project", type=str, default='rvgnn')


    #trick
    parser.add_argument('--supervised',default=True,action='store_true',help='Using Supervised Contrastive Learning')
    parser.add_argument('--balancedmask', default=False, action='store_true',help='Using balancedmask set for snn')
    parser.add_argument('--classcenter', default=True, action='store_true', help='Using classcenter support tensor for pseudo-labels')


    return parser.parse_known_args()[0]









def parse_args_sweep():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--datadir", type=str, default="/tmp/data")
    parser.add_argument("--dataset", type=str, default="Flickr", help="Cora, CiteSeer, PubMed, Computers, Photo,CS")

    # masking
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument('--imb_ratio', type=float, default=1, help='Imbalance Ratio')

    # Encoder
    parser.add_argument("--dim", type=int, default=128, help="64,128,256")
    parser.add_argument("--layers", type=int, default=2, help="1,2,3")
    parser.add_argument('--n_head', type=int, default=8,help='the number of heads in GAT')
    parser.add_argument('--net', type=str, default='GCN',help='GCN,GAT,SAGE,SGC,PPNP,CHEB')
    parser.add_argument('--chebgcn_para', type=int, default=2, help=' Chebyshev filter size of ChebConv')


    # optimization
    parser.add_argument("--epochs", '-e', type=int, default=2000, help="The number of epochs")
    parser.add_argument("--lr", '-lr', type=float, default=0.005, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--decay", type=float, default=1e-2, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--patience", type=int, default=200)

    # hyper-parameter
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--thres", type=float, default=0.8)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--lam2", type=float, default=2.5)

    # augmentation
    parser.add_argument("--df_1", type=float, default=0.5)
    parser.add_argument("--de_1", type=float, default=0.5)
    parser.add_argument("--df_2", type=float, default=0.2)
    parser.add_argument("--de_2", type=float, default=0.2)
    parser.add_argument("--device", type=int, default=0, help="GPU to use")

    # project(for logging)
    parser.add_argument("--project", type=str, default='rvgnn')
    # sweep id
    parser.add_argument("--sweep_id", type=str, default='')


    #trick
    parser.add_argument('--supervised',default=True,action='store_true',help='Using Supervised Contrastive Learning')
    parser.add_argument('--balancedmask', default=False, action='store_true',help='Using balancedmask set for snn')
    parser.add_argument('--classcenter', default=True, action='store_true', help='Using classcenter support tensor for pseudo-labels')

    return parser.parse_known_args()[0]




