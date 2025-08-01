# Common Utility Functions for training, testing, evaluating
import torch
from torch import nn
from .stock_dataset import StockDataset, StockDataset_V2, StockDataset_V3
from torch.utils.data import DataLoader
from .technical_analysis import get_data_set, get_data, toSequential_V2
import numpy as np
import os

#Calculate Utility based on policy Output
#z: z from dataset
#c: transaction cost
def calcUtility(policyOutput, z, c=0.0001):
    with torch.no_grad():
        discretize=policyOutput.detach()
        discretize = (discretize>0)*2-1
        preAction=torch.cat([discretize[:,0:1], discretize[:, :-1]], dim=1)
        #net income R
        R=z*discretize-c*((discretize-preAction)!=0)
        U=[]
        for i in range(R.shape[1]):
            if(i==0):
                u=R[:,i:i+1]
            else:
                u=R[:,i:i+1]+U[i-1]
            U.append(u)
        U=torch.cat(U, dim=1)
        return U, preAction

#Prevent exploding gradient
def grad_clipping(net, theta): 
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#model weight initialization
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.normal_(0.0,0.01)


#Loss function defined by paper
def lossFunc(predP, y, policyOutput, z, device: torch.device):
    #MSE
    term1=nn.MSELoss()(predP, y)
    #RL
    U, preAction=calcUtility(policyOutput, z)
    U_detach=U.detach()
    actionProb=(torch.tensor(1).to(device)+policyOutput)/torch.tensor(2)
    plusMinus=(preAction<0)*1
    term2=-torch.log(1*plusMinus+((-1)**plusMinus)*actionProb)*U_detach
    return term2.mean()+term1

    
#greedy loss function
def lossFunc2(predP, y, policyOutput, z, device):
    #MSE
    term1=nn.MSELoss()(predP, y)
    #RL
    greedyAction=(z>=0.01)*2.0-1.0
    U, preAction=calcUtility(policyOutput, z)
    U_detach=U.detach()
    actionProb=(torch.tensor(1).to(device)+policyOutput)/torch.tensor(2)
    plusMinus=(preAction<0)*1
    term2=(torch.log(1*plusMinus+((-1)**plusMinus)*actionProb)*U_detach).mean()
    term3=nn.MSELoss()(policyOutput, greedyAction)
    return term3+term2+term1

#Generation of training, validation, and testing dataset
def DataIterGen(test_id_list, val_id_list, name_list, full_list, demo=False, bsize=32) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    test_id_list: id of subjects for testing
    val_id_list: id of subjects for validation
    other subjects for training
    full_list=get_data_set(name_list), preprocessed
    demo: when demo mode is True, only test_iter is returned, with data from
    first entry of test_id_list (single stock)
    """
    name_count=len(name_list)

    if demo:
        test_iter=DataLoader(StockDataset(test_id_list[0:1], full_list, timestep=24, gap=1), shuffle=False, batch_size=64, num_workers=0)
        print(f'Demo with stock: {name_list[test_id_list[0]]} ')
        return test_iter
    else:
        train_list=set(name_list)-set([name_list[i] for i in test_id_list])-set([name_list[i] for i in val_id_list])
        train_count=len(train_list)
        partial_list=get_data_set(train_list)
        test_iter=DataLoader(StockDataset(test_id_list, full_list), batch_size=bsize, num_workers=0)
        val_iter=DataLoader(StockDataset(val_id_list, full_list), batch_size=bsize, num_workers=0)
        train_iter=DataLoader(StockDataset(list(range(train_count)), partial_list), shuffle=True, batch_size=bsize, num_workers=0)
        print(f'Val: {[name_list[val_id] for val_id in val_id_list]}, Test: {[name_list[test_id] for test_id in test_id_list]}, Train: {train_list} ')
        return train_iter, val_iter, test_iter

#Generation of training, validation, and testing dataset
def DataIterGen_V2(stock_id, name_list, demo=False, gap=1, window=24,
                   train_range: tuple = ("2013-01-01", "2017-10-01"),
                   test_range: tuple = ("2018-01-01", "2019-01-01"),
                   val_range: tuple = ("2017-10-08", "2018-04-01")):
    """
    stock_id: index in name_list
    demo: when demo mode is True, only test_iter is returned, with data from
    train_range: tuple of dates in ("YYYY-MM-DD", "YYYY-MM-DD") format
    ...

    returns:
    test_iter if demo
    ---
    train_iter, val_iter, test_iter if not demo
    """
    train_s, train_e = train_range
    test_s, test_e = test_range
    val_s, val_e = val_range
    print(f"Using periods\nTrain: {train_range}\nTest: {test_range}\nValidation: {val_range}")
    print("Initializing Training Dataset...", end='')
    trainDS=StockDataset_V2(stock_id, name_list, start=train_s, end=train_e, timestep=window)
    print("[DONE]")
    #get high correlation list for validation and testing
    hcl=trainDS.getHighCorrelationList()
    if demo:
        test_iter=DataLoader(StockDataset_V2(stock_id, name_list, timestep=window, gap=gap, 
                                        start=test_s, end=test_e,
                                        use_external_list=True, external_list=hcl), 
                           shuffle=False, batch_size=64, num_workers=0)
        #get abs change in stock closing price:
        data=get_data(name_list[stock_id], start=val_s, end=test_e)
        delta=data.iloc[-1]['Close']-data.iloc[91]['Close']
        print(f'Demo Stock ticker: {name_list[stock_id]}, change in closing price during testing period: ${delta:.2f}')
        return test_iter
    else:
        print("Initializing Iterators(dataloaders) From Dataset...",end='')
        test_iter=DataLoader(StockDataset_V2(stock_id, name_list, gap=gap, timestep=window,
                                        start=test_s, end=test_e,
                                        use_external_list=True, external_list=hcl),
                            batch_size=32, num_workers=0)
        val_iter=DataLoader(StockDataset_V2(stock_id, name_list, gap=gap, timestep=window,
                                       start=val_s, end=val_e,
                                       use_external_list=True, external_list=hcl),
                           batch_size=32, num_workers=0)
        train_iter=DataLoader(trainDS, shuffle=True, batch_size=32, num_workers=0)
        print("[DONE]")
        print(f'Stock ticker: {name_list[stock_id]}')
        return train_iter, val_iter, test_iter

#Generation of training, validation, and testing dataset
def DataIterGen_V2(stock_id, name_list, demo=False, gap=1, window=24, bsize=32,
                   train_range: tuple = ("2013-01-01", "2017-10-01"),
                   test_range: tuple = ("2018-01-01", "2019-01-01"),
                   val_range: tuple = ("2017-10-08", "2018-04-01")):
    """
    stock_id: index in name_list
    demo: when demo mode is True, only test_iter is returned, with data from
    train_range: tuple of dates in ("YYYY-MM-DD", "YYYY-MM-DD") format
    ...

    returns:
    test_iter if demo
    ---
    train_iter, val_iter, test_iter if not demo
    """
    train_s, train_e = train_range
    test_s, test_e = test_range
    val_s, val_e = val_range
    name = name_list[stock_id]
    if demo:
        test_iter=DataLoader(StockDataset_V3(*get_cache_or_generate(stock_id, name_list, test_s, test_e, gap, window), stock_id), shuffle=False, batch_size=bsize)
        print(f'Demo Stock ticker: {name}, testing period: {test_s} to {test_e} [YYYY-MM-DD]')
        return test_iter

    # Not demo, so ge training testing and validation sets
    print(f"Using periods\nTrain: {train_range}\nTest: {test_range}\nValidation: {val_range}")
    print("Initializing Training Dataset...", end='')
    trainDS = StockDataset_V3(*get_cache_or_generate(stock_id, name_list, train_s, train_e, gap, window), stock_id)
    hcl = trainDS.getHighCorrelationList()

    print("Initializing Iterators(dataloaders) From Dataset...",end='')
    test_iter=DataLoader(StockDataset_V3(*get_cache_or_generate(stock_id, name_list, test_s, test_e, gap, window, ext_hcl=hcl), stock_id), batch_size=bsize, num_workers=0)
    val_iter=DataLoader(StockDataset_V3(* get_cache_or_generate(stock_id, name_list, val_s, val_e,   gap, window, ext_hcl=hcl), stock_id), batch_size=bsize, num_workers=0)
    train_iter=DataLoader(trainDS, shuffle=True, batch_size=32, num_workers=0)

    print("[DONE]")
    print(f'Stock ticker: {name_list[stock_id]}')
    return train_iter, val_iter, test_iter

def get_cache_or_generate(stock_id, name_list, start, end, gap, window, ext_hcl=None):
    name = name_list[stock_id]
    if checkCache(name, start, end, gap, window):
        X, y, z, zp, hcl = [np.load(path) for path in checkCache(name, start, end, gap, window)]
    else:
        print('Generating')
        if ext_hcl is not None: X, y, z, zp, hcl = toSequential_V2(stock_id, name_list, timeStep=window, gap=gap, start=start, end=end, use_external_list=True, external_list=ext_hcl)
        else:                   X, y, z, zp, hcl = toSequential_V2(stock_id, name_list, timeStep=window, gap=gap, start=start, end=end, use_external_list=False)
        for tag, item in zip(('X', 'y', 'z', 'zp', 'hcl'), (X, y, z, zp, hcl)):
            savedir = '_'.join([name, start, end, str(gap), str(window)])
            os.makedirs(f'.cache/{savedir}', exist_ok=True)
            np.save(os.path.join('.cache', savedir, tag), arr=item, allow_pickle=True)
    if ext_hcl is not None:
        return X, y, z, zp, ext_hcl
    return X, y, z, zp, hcl

def checkCache(*args):
    if '.cache' not in os.listdir():
        os.mkdir('.cache')
        return None
    if '_'.join([str(arg) for arg in args]) in os.listdir('./.cache'):
        pathname = '_'.join([str(arg) for arg in args]) 
        return ['./.cache/' + pathname + '/' + tag + '.npy' for tag in ('X', 'y', 'z', 'zp', 'hcl')]
    return None