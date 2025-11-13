"""
存放用于训练、测试模型的函数
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


def forward(model: nn.modules,
            state: torch.Tensor, point, point_type, edge, times, conds, device='cuda', mode="train"):
    predicts, deltas = [], []

    state, point, edge = state.to(device), point.to(device), edge.to(device)
    times, conds = times.to(device), [cond.to(device) for cond in conds]
    point_type = point_type.to(device)

    state_in = state[:, 0] # 输入状态

    predicts.append(state_in)
    
    for t in range(state.shape[1]-1):
        next_state, delta_state = model(point, point_type, edge, times[:,t], predicts[t].detach(), conds) # node_type.float())

        # 边界条件的强制执行

        predicts.append(next_state)
        deltas.append(delta_state)
    
    predicts = torch.stack(predicts[1:], dim=1) # 把预测出来的进行合并
    deltas = torch.stack(deltas, dim=1)

    return predicts, deltas

def get_l2_loss(output, target, ep=1e-8):
    # output.dim = (batch, seq, N, c) or (batch, seq, N)
    # target.dim = (batch, seq, N, c) or (batch, seq, N)
    target = target[:, 1:]
    if output.dim() == 4:
        if output.shape[-1] == 1:
            output = output.squeeze(-1) 
            target = target.squeeze(-1) 
            error = (output - target)
            norm_error = torch.norm(error, dim=-1) / (torch.norm(target, dim=-1) + ep)
            norm_error_time = torch.mean(norm_error, dim=-1)
            norm_error_batch = torch.mean(norm_error_time, dim=0)
        else:
            error = (output - target)
            norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + ep)
            norm_error_channel = torch.mean(norm_error, dim=-1)
            norm_error_time = torch.mean(norm_error_channel, dim=-1)
            norm_error_batch = torch.mean(norm_error_time, dim=0)
    elif output.dim() == 3:
        error = (output - target)
        norm_error = torch.norm(error, dim=-1) / (torch.norm(target, dim=-1) + ep)
        norm_error_time = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_time, dim=0)
    return norm_error_batch

def get_each_l2(predict_hat, label_gt): # 计算每个时间点的平均误差
    t_step = label_gt.shape[1] - 1
    losses_each_t = torch.zeros(t_step)
    for t in range(t_step):
        error = predict_hat[:,t] - label_gt[:,t+1]
        norm_error = torch.norm(error, dim=-2) / (torch.norm(label_gt[:,t], dim=-2) + 1e-6)
        norm_error_channel = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_channel, dim=0)
        losses_each_t[t] = norm_error_batch.item()
    return losses_each_t

def get_train_loss(predict: torch.Tensor, 
                   delta: torch.Tensor, 
                   state: torch.Tensor, mask, loss_flag='L2_loss'):
    assert loss_flag == 'MGN_norm_loss' or loss_flag == 'L2_loss' 

    device = predict.device
    state = state.to(device)
    mask = mask.unsqueeze(1).unsqueeze(-1).to(device) # [B, 1, N, 1]
    losses = {}

    if loss_flag == "MGN_norm_loss":
        losses['loss'] = nn.MSELoss(delta * mask, (state[:,1:]-state[:,:-1])*mask)
    
    elif loss_flag == "L2_loss":
        losses['loss'] =  get_l2_loss(predict * mask, state * mask)

    # state = state[:, 1:]
    losses['L2_T'] = get_l2_loss(predict[...,0] * mask[...,0], state[...,0] * mask[...,0]).item()
    losses['each_l2'] = get_each_l2(predict * mask, state * mask)
    return losses

def get_val_loss(predict:torch.Tensor, state, mask, loss_flag='L2loss'):
    device = predict.device
    state = state.to(device)
    mask = mask.unsqueeze(1).unsqueeze(-1).to(device) # [B, 1, N, 1]
    losses = {}
    
    losses['L2_T'] = get_l2_loss(predict[...,0] * mask[...,0], state[...,0] * mask[...,0])
    losses['each_l2'] = get_each_l2(predict * mask, state * mask)
    return losses

def train(model: nn.modules,
          dataloader,
          optim, device='cuda', horizon=1):
    
    all_batch_num = 0 # 用于计算平均误差
    L2_T, each_l2 = 0, torch.zeros(horizon-1)
    for i, input in enumerate(dataloader):
        state, point, edge = input['state'], input['point'], input['edge']
        times, mask = input['time'], input['mask']
        conds = [input['thermal'], input['material'], input['dump']]
        point_type = input['point_type']
        batch_num = state.shape[0]
        predict, delta = forward(model, state, point, point_type, edge, times, conds, device, mode='train')
        costs = get_train_loss(predict, delta, state, mask)
        costs['loss'].backward()
        optim.step()
        optim.zero_grad()
        all_batch_num += batch_num
        L2_T += costs['L2_T'] * batch_num
        each_l2 += costs['each_l2'] * batch_num

    avg_error = {}
    avg_error['L2_T'] = L2_T / all_batch_num
    avg_error['each_l2'] = each_l2 / all_batch_num
    return avg_error

def validate(model: nn.modules,
             data_loader,
             device: torch.device='cuda', horizon=8):
    L2_T, each_l2 = 0, torch.zeros(horizon-1)
    all_batch_num = 0
    with torch.no_grad():
        # need inference
        for i, input in enumerate(data_loader):
            state, point, edge = input['state'], input['point'], input['edge']
            times, mask = input['time'], input['mask']
            conds = [input['thermal'], input['material'], input['dump']]
            point_type = input['point_type']

            batch_num = state.shape[0]

            predict, _ = forward(model, state, point, point_type, edge, times, conds, device, mode='test')

            costs = get_val_loss(predict, state, mask)
            all_batch_num += batch_num

            L2_T += costs['L2_T'] * batch_num
            each_l2 += costs['each_l2'] * batch_num
    avg_error = {}
    avg_error['L2_T'] = L2_T / all_batch_num
    avg_error['each_l2'] = each_l2 / all_batch_num
    return avg_error
