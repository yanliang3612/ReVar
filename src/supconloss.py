from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F



def supconloss(anchor_rep, pos_rep,mask,label,n_cls):
    loss_labeled = 0.
    N = 0.
    for m in range(n_cls):
        m_mask = torch.logical_and(label == m, mask)
        m_anchor_rep = anchor_rep[m_mask]
        m_pos_rep= pos_rep[m_mask]
        m_rep = torch.cat((m_anchor_rep, m_pos_rep), 0)
        l = m_rep.shape[0]
        N += l*(l-1)
        for i in range(l):
            loss_full = F.cosine_similarity(m_rep[i].repeat(l,1),m_rep,dim=-1)
            loss_full[i] = 0
            loss_labeled += (loss_full.sum())
    loss_unlabeled = F.cosine_similarity(anchor_rep[~mask],pos_rep[~mask],dim=-1).mean()
    loss = 2- 2 * (loss_labeled/(N) + loss_unlabeled)

    return loss



def supconloss2(anchor_rep, pos_rep,mask,label,n_cls):
    loss_labeled = 0.
    N = 0.
    for m in range(n_cls):
        m_mask = torch.logical_and(label == m, mask)
        m_anchor_rep = anchor_rep[m_mask]
        m_pos_rep= pos_rep[m_mask]
        m_rep = torch.cat((m_anchor_rep, m_pos_rep), 0)
        # shape [N, N]
        l = m_rep.shape[0]
        N += l * (l-1)
        m_cos_similarity = F.cosine_similarity(m_rep[None, :, :], m_rep[:, None, :], dim=-1)
        not_eye = 1 - torch.eye(m_cos_similarity.shape[0], requires_grad=False, device=m_cos_similarity.device)
        m_cos_similarity *= not_eye  # [N, N]
        loss_labeled += (m_cos_similarity.sum())

    loss_unlabeled = F.cosine_similarity(anchor_rep[~mask],pos_rep[~mask],dim=-1).mean()
    loss = 2- 2 * (loss_labeled/(N) + loss_unlabeled)

    return loss