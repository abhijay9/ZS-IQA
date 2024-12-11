import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from ot.lp import wasserstein_1d

def KL_div(p_output, q_output, get_softmax=True, epsilon=1e-10):
    KLDivLoss = nn.KLDivLoss(reduction='none')
    if get_softmax:
        p_output = F.softmax(p_output, dim=0) + epsilon
        q_output = F.softmax(q_output, dim=0) + epsilon
    
    # Ensure no values are below epsilon
    p_output = torch.clamp(p_output, min=epsilon)
    q_output = torch.clamp(q_output, min=epsilon)

    # Compute logs after clamping
    p_log = torch.log(p_output)
    q_log = torch.log(q_output)
    
    # Use KLDivLoss directly on clamped and logged outputs
    part1 = KLDivLoss(q_log, p_output).sum(dim=0)
    part2 = KLDivLoss(p_log, q_output).sum(dim=0)

    part1[part1 == float('inf')] = 0
    part2[part2 == float('inf')] = 0

    return (part1+part2)/2

def KL_distance(X, Y, win=8):
    chn_num = X.shape[1]
    X_patch = torch.reshape(X, [win, win, chn_num, -1])
    Y_patch = torch.reshape(Y, [win, win, chn_num, -1])
    patch_num = (X.shape[2] // win) * (X.shape[3] // win)

    X_1D = torch.reshape(X_patch, [-1, chn_num * patch_num])
    Y_1D = torch.reshape(Y_patch, [-1, chn_num * patch_num])

    X_1D_pdf = X_1D
    Y_1D_pdf = Y_1D

    kld = KL_div(X_1D_pdf, Y_1D_pdf)

    # import pdb; pdb.set_trace()

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w = (1 / (torch.sqrt(torch.exp((- 1 / (kld + 10)))) * (kld + 10) ** 2))

    final = kld + L2 * w

    return final.mean()
    # return final.nanmean()

def ws_distance(X,Y,P=1,win=8,device='cuda'):

    chn_num = X.shape[1]

    X_patch   = torch.reshape(X,[win,win,chn_num,-1]).type(torch.cuda.FloatTensor)
    Y_patch   = torch.reshape(Y,[win,win,chn_num,-1]).type(torch.cuda.FloatTensor)
    patch_num = (X.shape[2]//win) * (X.shape[3]//win)

    X_1D = torch.reshape(X_patch,[-1,chn_num*patch_num])
    Y_1D = torch.reshape(Y_patch,[-1,chn_num*patch_num])

    interval = np.arange(0, X_1D.shape[0], 1)
    all_samples = torch.from_numpy(interval).to(device).repeat([patch_num*chn_num,1]).t()

    X_pdf = F.softmax(X_1D, dim=-1)
    Y_pdf = F.softmax(Y_1D, dim=-1)

    wsd   = wasserstein_1d(all_samples, all_samples, X_pdf, Y_pdf, P)

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w  =  (1 / ( torch.sqrt(torch.exp( (- 1/(wsd+10) ))) * (wsd+10)**2))

    final = wsd + L2 * w
    # final = wsd

    return final.mean()

def cos_dist(feats_dis, feats_ref, add_inf_handling=False):
    if add_inf_handling:
        similarity = (1 - F.cosine_similarity(feats_dis, feats_ref, dim=-1))
        similarity[torch.isinf(similarity)] = 0.0
        return similarity.nanmean().item()
    else:
        return (1 - F.cosine_similarity(feats_dis, feats_ref, dim=-1)).mean().item()

def l2(feats_dis, feats_ref):
    return ((feats_dis-feats_ref)**2).mean().item()

SCALING_FACTOR = 1
def swd_dist(feats_dis, feats_ref, device, Ndirection = 20):
    b, dim, h, w = feats_ref.shape
    n = h*w

    feats_dis = feats_dis.view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR).float()
    feats_ref = feats_ref.view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR).float()

    # sample random directions
    Ndirection = dim

    directions = torch.randn(Ndirection, dim).to(device=device)
    directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))

    # project activations over random directions
    projected_activations_ref = torch.einsum('bdn,md->bmn', feats_dis, directions)
    projected_activations_dis = torch.einsum('bdn,md->bmn', feats_ref, directions)

    # sort the projections
    sorted_activations_ref = torch.sort(projected_activations_ref, dim=2)[0]
    sorted_activations_dis = torch.sort(projected_activations_dis, dim=2)[0]

    # L2 over sorted lists
    return ((sorted_activations_ref-sorted_activations_dis)**2).mean(dim=1).cpu().numpy()[0]

def js_div(p_output, q_output, get_softmax=True, epsilon=1e-10):
    KLDivLoss = nn.KLDivLoss(reduction='none')
    if get_softmax:
        p_output = F.softmax(p_output, dim=0) + epsilon
        q_output = F.softmax(q_output, dim=0) + epsilon
    
    # Ensure no values are below epsilon
    p_output = torch.clamp(p_output, min=epsilon)
    q_output = torch.clamp(q_output, min=epsilon)

    log_mean_output = ((p_output + q_output )/2 + 1e-17).log()
    part1 = KLDivLoss(log_mean_output, p_output).sum(dim=0)
    part2 = KLDivLoss(log_mean_output, q_output).sum(dim=0)

    part1[part1 == float('inf')] = 0
    part2[part2 == float('inf')] = 0
    
    return (part1+part2)/2

def js_distance(X, Y, P=2, win=8):
    chn_num = X.shape[1]

    X_patch = torch.reshape(X, [win, win, chn_num, -1])
    Y_patch = torch.reshape(Y, [win, win, chn_num, -1])
    patch_num = (X.shape[2] // win) * (X.shape[3] // win)

    X_1D = torch.reshape(X_patch, [-1, chn_num * patch_num])
    Y_1D = torch.reshape(Y_patch, [-1, chn_num * patch_num])

    X_1D_pdf = X_1D
    Y_1D_pdf = Y_1D

    jsd = js_div(X_1D_pdf, Y_1D_pdf)

    final = jsd

    return final.mean()