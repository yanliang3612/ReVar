import torch


def classcenter(num_classes,anchor_rep,pos_rep,labels,train_mask,device):
    anchor_centroid_list = []
    pos_centroid_list = []
    for m in range(num_classes):
        m_mask = torch.logical_and(labels == m, train_mask)
        m_anchor_rep = anchor_rep[m_mask]
        m_pos_rep = pos_rep[m_mask]
        m_anchor_centroid = m_anchor_rep.mean(0)
        m_pos_centroid = m_pos_rep.mean(0)
        anchor_centroid_list.append(m_anchor_centroid)
        pos_centroid_list.append(m_pos_centroid)
    anchor_centroid = torch.stack(anchor_centroid_list).to(device)
    pos_centroid = torch.stack(pos_centroid_list).to(device)
    matrix = torch.eye(num_classes).to(device)

    return anchor_centroid,pos_centroid,matrix

