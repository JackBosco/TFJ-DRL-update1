import os
from typing import Tuple, List, Type
import imageio
import numpy as np
from .utils import init_weights, lossFunc2, grad_clipping, calcUtility
import torch.optim as optim
import matplotlib.pyplot as plt
import torch


def train_posthoc(train_iter, eval_iter, net, optimizer1, optimizer2, device, num_epoch, name, lossfn=lossFunc2) -> Tuple[np.array, np.array]:
    loss_data=[]
    U_data=[]
    conf_data = np.array([])
    image_files = []
    os.makedirs('training_pics', exist_ok=True)
    net.apply(init_weights)
    net.to(device)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer1, 0.95, last_epoch=-1)

    for epoch in range(num_epoch):
        net.train()
        lossEpoch, lossEpoch_list, conf_list =train_epoch_posthoc(train_iter, net, device, optimizer1, optimizer2, lossfn=lossfn)   
        loss_v, U_v, embeds, confusion=prediction_posthoc(eval_iter, net, device, lossfn=lossfn)
        loss_data.append(lossEpoch)  
        U_data.append(U_v)
        print(f'Epoch {epoch}, training loss: {lossEpoch:.2f}, val utility: {U_v:.2f}, confusion: {confusion:.3f}')
        plt.figure()
        x, y = embeds[:, 0], embeds[:, 1]
        plt.scatter(x,y, alpha=0.25)
        image_file = f"epoch_{epoch + 1}.png"
        plt.savefig('training_pics/'+image_file)
        plt.close()
        image_files.append(image_file)
        torch.save(net.state_dict(), os.path.join('./model_weights', f'{name}-epoch-{epoch+1}.pth'))
        scheduler.step()
        conf_data = np.append(conf_data, conf_list)
    
     
    #plot loss & Utility
    fig, ax_left = plt.subplots(figsize=(10,4))
    ax_right = ax_left.twinx()
    ax_left.plot(loss_data, label = "Loss", color='blue')
    ax_right.plot(U_data, label = "Utility", color='red')
    ax_left.set_xlabel('Time Step')
    ax_right.set_ylabel('Loss y axis')
    ax_right.set_ylabel('Utility y axis')
    ax_left.set_title('Loss and Utility')
    ax_left.legend()
    ax_right.legend()
    return loss_data, conf_data

def train_epoch_posthoc(train_iter, net, device, optim1, optim3, lossfn):
    loss_data=[]
    embed_list = []
    confusion_list = np.array([])
    with torch.autograd.set_detect_anomaly(True):
        for X, y, z, _ in train_iter:
            # reset state for each batch
            state= net.begin_state(batch_size=X.shape[0], device=device)
        
            # move to device
            X, y, z=X.to(device), y.to(device), z.to(device)
            
            # feed through full-monty
            predP, output, (confusion_granular, confusion, embed) = net(X, state)
            
            # ============[ ADDING DECODER LOSS CONFUSION, DREAM] =====================
            optim3.zero_grad()
            confusion.backward()
            optim3.step()
            # =========================================================================

            loss = lossfn(predP, y, output,z, device)
            optim1.zero_grad()
            loss.backward()
            grad_clipping(net, 1)
            optim1.step()
            loss_data.append(loss.item())
            confusion_list = np.append(confusion_list, confusion_granular.flatten().cpu())
    return np.array(loss_data).mean(), loss_data, confusion_list

def apply_conf(policy: List[List], confusion: List[List], lower_thresh=0.15, upper_thresh=0.85) -> None:
    """
    confusion:  [BATCH, STEP]
    policy:     [BATCH, STEP]
    """
    x = confusion.mean(dim=-1)
    x = ((x < upper_thresh)*1) * ((x > lower_thresh)*1)
    policy[:,-1] *= x
    return policy

def prediction_posthoc(eval_iter, net, device, lossfn):
    net.eval()
    net.conf_net.eval()
    loss_list=[]
    U_list=[]
    embed_list=[]
    conf_list=np.array([])
    with torch.no_grad():
        for X, y, z, _ in eval_iter:
            # to device
            X, y, z = X.to(device), y.to(device), z.to(device)

            # initialize rnn state
            state=net.begin_state(batch_size=X.shape[0], device=device) # changed from net.begin

            # feed through full-monty
            predP, output, (confusion,_, embed) = net(X, state)
            
            # ==========={ HEURISTIC: APPLYING BOUNDS [0.15, 0.85] }========
            bsize = X.shape[0]
            policy = apply_conf(output, confusion)
            output = policy.reshape(output.shape) 
            # ==============================================================

            loss=lossfn(predP, y, output, z, device).float()
            U, _=calcUtility(output, z)
            loss_list.append(loss.cpu().numpy())
            U_list.append(U[:, -1].mean().cpu().numpy())
            embed_list.append(embed.flatten(0,1).cpu().numpy())
            conf_list = np.concatenate((conf_list, confusion.flatten().cpu().numpy()))
    return np.array(loss_list).mean(), np.array(U_list).mean(), np.concatenate(embed_list), conf_list.mean()

def test_posthoc(net, test_iter, device, epoch, name):
    net.eval()
    net.conf_net.eval()
    net.load_state_dict(torch.load(os.path.join('./model_weights', f'{name}-epoch-{epoch}.pth')))
    net.to(device)

    U_list=[]
    conf_list=np.array([])
    with torch.no_grad():
        for X, _, _, zp in test_iter:
            X, zp = X.to(device),  zp.to(device)
            state=net.begin_state(batch_size=X.shape[0], device=device)
            predP, output, (confusion, _, embed) = net(X, state)
            U, _=calcUtility(output, zp)
            U_list.append(U[:, -1].mean().cpu().numpy())

            #== [confusion] ==
            conf_list = np.append(conf_list, confusion.flatten().cpu().numpy())
            # ===============
    return np.array(U_list).mean(), conf_list.mean()

def demo_posthoc(net: torch.nn.Module, demo_iter:torch.utils.data.DataLoader, device: torch.device, epoch, name,
         title='TFJ-DRL With Confusion', conf_bounds = (0.15, 0.85)) -> Tuple[np.ndarray,np.ndarray, np.ndarray, np.ndarray]:
    net.eval()
    net.conf_net.eval()
    net.load_state_dict(torch.load(os.path.join('./model_weights', f'{name}-epoch-{epoch}.pth')))
    net.to(device)
    
    reward=np.array([0])
    reward_noconf=np.array([0])
    conf_list=np.array([])
    stock_change=np.array([0])
    with torch.no_grad():
        for X, _, _, zp in demo_iter:
            X, zp = X.to(device),  zp.to(device)
            state=net.begin_state(batch_size=X.shape[0], device=device)
            predP, output, (confusion, _, embed) = net(X, state)
            discretize=output.detach()
            discretize = (discretize>0)*2-1
            disc_noconf = discretize.clone()*zp
            # ==========={ HEURISTIC: APPLYING BOUNDS [0.15, 0.85] }========
            bsize, step = X.shape[:2]
            discretize = apply_conf(discretize.view(bsize, -1), confusion.view(bsize, -1), conf_bounds[0], conf_bounds[1])
            discretize = discretize.reshape(output.shape)
            # ==============================================================
            batchReward=discretize*zp
            reward_noconf=np.concatenate((reward_noconf,disc_noconf[:,-1].reshape(-1).cpu().numpy()))
            reward=np.concatenate((reward,batchReward[:,-1].reshape(-1).cpu().numpy()))
            stock_change=np.concatenate((stock_change, zp[:,-1].reshape(-1).cpu().numpy()))
            
            #== [confusion] ==
            conf_list = np.concatenate((conf_list, confusion.reshape(X.shape[0], -1).mean(dim=-1).cpu().numpy()))
            # ===============
            
        result = [sum(reward[ : i + 1]) for i in range(len(reward))] 
    return result, conf_list, stock_change, np.cumsum(reward_noconf)

def show_demo_posthoc(result, conf_list, stock_change, baseline, conf_bounds):
    fig, [underlying, ax_left, ax_right] = plt.subplots(3, figsize=(20,12), sharex=True)
    if baseline is not None:
        ax_left.plot(baseline, label = "TFJ-DRL Base", color='brown')
        ax_left.set_ylabel("Cumulative Gain")
        ax_left.legend()
    ax_left.plot(result, label = "TFJ-DRL With Confusion", color='blue')
    ax_left.hlines([0], 0, 1, transform=ax_left.get_yaxis_transform(), linestyle='dashed', colors='gray', label='Initial Gain')
    ax_left.set_xlabel('Time Step')
    ax_left.set_ylabel('Cumulative Gain')
    ax_right.set_ylabel('Confusion')
    ax_right.plot(conf_list, label='Confusion at Time Step')
    ax_right.hlines(conf_bounds, 0, 1, transform=ax_right.get_yaxis_transform(), linestyle='dashed', colors=('red', 'green'), label='Allowable Confusion Boundaries')
    ax_left.set_title('TFJ-DRL')
    underlying.set_title('Buy and Hold')
    underlying.plot(np.cumsum(stock_change), label="Change in Closing Price", color='blue')
    underlying.hlines([0], 0, 1, transform=underlying.get_yaxis_transform(), linestyle='dashed', colors='gray', label='Initial Gain')
    ax_left.legend()
    ax_right.legend()
    underlying.legend()
    return fig, (underlying, ax_left, ax_right)