# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
from einops import rearrange


def naive_osgm_net_decoding_one_step(q, k, v, g, lamb, beta, prev_h_kk, prev_h_kv):
    q = q.float().clone()
    k = k.float().clone()
    v = v.float().clone()
    g = g.float().clone()
    lamb = lamb.float().clone()
    beta = beta.float().clone()
    B, h, d = q.shape
    k_beta = k * beta.unsqueeze(-1)

    h_kk = prev_h_kk * g.exp()[..., None] + k_beta * k
    h_kv = prev_h_kv * g.exp()[..., None, None] + k_beta.unsqueeze(-1) * v.unsqueeze(-2)
    lamb = lamb.unsqueeze(0)
    q_final = q / (h_kk + lamb)
    o = (q_final.unsqueeze(-1) * h_kv).sum(-2)
    return o, h_kk, h_kv


def naive_osgm_net(q, k, v, g, lamb, beta, h_kk_init=None, h_kv_init=None):
    B, L, h, d = q.shape
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    lamb = lamb.float()
    beta = beta.float()

    h_kk = h_kk_init.clone() if h_kk_init is not None else torch.zeros(B, h, d, device=q.device)
    h_kv = h_kv_init.clone() if h_kv_init is not None else torch.zeros(B, h, d, d, device=q.device)

    h_kk_all = torch.zeros(B, L, h, d, device=q.device)
    h_kv_all = torch.zeros(B, L, h, d, d, device=q.device)
    for i in range(L):
        h_kk = h_kk * g[:, i, :, None].exp() + k[:, i, :, :] ** 2 * beta[:, i, :, None] 
        h_kv = h_kv * g[:, i, :, None, None].exp() + (k[:, i, :, :] * beta[:, i, :, None]
                                                      )[..., None] * v[:, i, :, None, :]
        h_kk_all[:, i] = h_kk
        h_kv_all[:, i] = h_kv

    q_star_gold = q / (h_kk_all + lamb[None, None, ...,])
    o_gold = (q_star_gold[..., :, None] * h_kv_all).sum(-2)
    return o_gold, h_kk, h_kv

if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, h, d = 2, 5, 4, 8
    q = torch.randn(B, L, h, d)
    k = torch.randn(B, L, h, d)
    v = torch.randn(B, L, h, d)
    g = torch.randn(B, L, h)
    beta = torch.randn(B, L, h).abs()
    lamb = torch.randn(d).abs()          # 长度=d，加在对角线上

    o, h_kk_final, h_kv_final = naive_osgm_net(q, k, v, g, lamb, beta)
    o, h_kk_final_dec, h_kv_final_dec = naive_osgm_net_decoding_one_step(
        q[:, -1], k[:, -1], v[:, -1], g[:, -1], lamb, beta[:, -1], 
        prev_h_kk=torch.zeros(B, h, d), prev_h_kv=torch.zeros(B, h, d, d)
    )

    print("output o shape :", o.shape)               # 应为 (B,L,h,d)
    print("final h_kk shape:", h_kk_final.shape)     # (B,h,d)
    print("final h_kv shape:", h_kv_final.shape)     # (B,h,d,d)
    print("o min/max:", o.min().item(), o.max().item())