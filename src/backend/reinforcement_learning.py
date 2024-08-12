import torch
from torch import nn
import numpy as np

# Define GRU (Gated Recurrent Unit) class, inheriting from nn.Module
class GRU(nn.Module):
    # Initialization method for the GRU class
    # env_size: size of the environment, used in reinforcement learning (RL) algorithm
    def __init__(self, env_size):
        super(GRU, self).__init__()
        # Define a single-layer GRU with input size 86, hidden size 128, and batch first
        # `input_size=86`: Input feature dimension
        # `hidden_size=128`: Number of features in the hidden state
        # `num_layers=1`: Number of recurrent layers (single layer in this case)
        # `batch_first=True`: Input and output tensors are provided as (batch, seq, feature)
        self.rnn = nn.GRU(
            input_size=86,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        # Define two Linear layers
        # `self.linear1`: Processes concatenated GRU states and temporal attention vector
        # `self.linear2`: Final linear layer, producing the predicted output
        self.linear1 = nn.Linear(256, env_size)
        self.linear2 = nn.Linear(env_size, 1)
        self.num_layers = 1

    def forward(self, x, state, device):
        """
        Summary:
            Forward pass for the GRU model

        Args:
            x: input tensor
            state: hidden state tensor
            device: device on which to perform computations (CPU or CUDA)
        
        Returns:
            GRU output before the final linear transformation, after the final linear transformation, and the current state from the RNN.
        """
        # Get batch size and timestep from input's shape
        batch_size, timestep, _ = x.shape
        # Process input through the GRU
        states, state = self.rnn(x, state)
        # Apply temporal attention mechanism
        tamVec = tam(states, device)

        # Concatenate GRU states and temporal attention vectors
        # Shape: (batch_size, time_step, hidden_size*2)
        concatVec = torch.cat([states, tamVec], axis=2)
        # Linear transformation followed by Tanh activation
        envVec = self.linear1(torch.tanh(concatVec.reshape(batch_size * timestep, -1)))
        # Apply Dropout and ReLU activation
        output = nn.Dropout(p=0.4)(envVec)
        output = nn.ReLU()(output)
        # Final linear transformation
        output = self.linear2(output)
        # Reshape the envVec and output
        envVec = envVec.reshape(batch_size, timestep, -1)
        output = output.reshape(batch_size, timestep, -1)
        return (output.view(batch_size, -1), envVec), state

    # Initialize the hidden state for the GRU
    def begin_state(self, device, batch_size=1):
        # `nn.GRU` takes a tensor as hidden state
        return torch.zeros((
            self.rnn.num_layers, batch_size, 128), device=device)

# Temporal attention mechanism calculation
def tam(states: torch.Tensor, device: torch.device):
    """
    Temporal Attention Mechanism (TAM)
    Args:
        states: Tensor of shape (batch_size, time_step, hidden_size)
        device: Device on which the tensor computations should be performed
    
    Returns:
        Tensor of shape (batch_size, time_step, hidden_size*2)
    """
    with torch.no_grad():
        # Get batch size, time step, and hidden size from states' shape
        b_size, t_step, h_size = states.shape
        # Initialize the temporal attention vector list
        vec_list = torch.tensor(np.zeros([b_size, 1, h_size]).astype('float32')).to(device)

        # Iterate through each time step starting from 1
        for i in range(1, states.shape[1]):
            # Initialize cumulative and batch attention vectors
            batchedCum = torch.tensor(np.zeros([b_size, 1, 1]).astype('float32')).to(device)
            batch_list = []
            vec = torch.tensor(np.zeros([b_size, 1, h_size]).astype('float32')).to(device)
            # Compute attention weights for each previous time step
            for j in range(i):
                batched = torch.exp(torch.tanh(torch.bmm(states[:, i:i+1, :], torch.transpose(states[:, j:j+1, :], 1, 2))))
                batch_list.append(batched)
                batchedCum += batched
            # Aggregate vectors using attention weights
            for j in range(i):
                vec += torch.bmm((batch_list[j] / batchedCum), states[:, j:j+1, :])
            # Append the computed vector to the list
            vec_list = torch.cat([vec_list, vec], axis=1)
    # Return the list of vectors
    return vec_list


def rlForwardFunc(envs, params):
    """
    Summary:
        RL forward function to compute actions
    Args:
        envs: States output by the GRU (shape: batch_size, num_steps, envs_size)
        params: Parameters initialized by get_params()

    Returns:
        Tensor of computed actions (shape: batch_size, num_steps, env_size)
    """
    W, b, h = params
    outputs = []
    tanh=nn.Tanh()
    # Shape of `X`: (`batch_size`, `envs_size`)
    for i in range(envs.shape[1]):
        X = envs[:,i,:]
        Y = torch.matmul(X, W) + b
        Z1 = tanh(Y)
        Z = Z1.clone()
        Z=Z.unsqueeze(1)
        
       
        if(i==0):
            outputs.append(Z)
        else:
            Z+=outputs[i-1]* h
            outputs.append(Z)
    return torch.cat(outputs, dim=1)

class rlPolicy(nn.Module):
    """RL Policy net modeled by parameters"""
    def __init__(self, env_size, device: torch.device):
        super(rlPolicy, self).__init__()
        
        #self.linear = nn.Linear(32+1, 1)
        
        W, b, h=get_params(env_size, device)
        self.device=device
        self.W=nn.Parameter(W)
        self.b=nn.Parameter(b)
        self.h=nn.Parameter(h)
        self.rnn=GRU(env_size)

    # Forward pass through the RL Policy network
    def forward(self, x, state):
        # Obtain predictions and environment vector from GRU
        (predP, envVec), state = self.rnn(x, state, self.device)
        
        # Compute actions based on the environment vector and parameters
        output = rlForwardFunc(envVec, [self.W, self.b, self.h])
        
        # Return predictions and actions
        return predP, output
    
    # Initialize the hidden state for the RL Policy network
    def begin_state(self, device, batch_size=1):
        return self.rnn.begin_state(device, batch_size)

# Customly initialize parameters for RL model
def get_params(env_size: int, device: torch.device):
    """
    Args:
        env_size (int): Size of the environment
        device (torch.device): Device to initialize the parameters on

    Returns:
        Tuple (W, b, h) representing the model parameters
    """
    num_inputs = env_size

    # Function to initialize weights with a normal distribution
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.03

    # Initialize parameters W, b, and h with required gradients
    W = normal((num_inputs,))
    b = torch.zeros(1, device=device)
    h = torch.randn(1, device=device) * 0.01
    # Output layer parameters
    # Attach gradients
    # params = [W, b, h]
    # for param in params:
    #     param.requires_grad_(True)
    #     param=nn.Parameter(param)
    W.requires_grad_(True)
    b.requires_grad_(True)
    h.requires_grad_(True)
    return W, b, h