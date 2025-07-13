import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict,List,Optional,Tuple
from torch.utils.data import DataLoader,TensorDataset

class ANN(nn.Module):
    def __init__(self,inputSize: int,hiddenSize:int,outputSize: int):
        super().__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        self.fc1 = nn.Linear(inputSize,hiddenSize)
        self.fc2 = nn.Linear(hiddenSize,hiddenSize)
        self.fc3 = nn.Linear(hiddenSize,outputSize)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
       
    
    def forward(self,x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
    
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
    
        x = self.fc3(x)

        return x
        
    def getArchitectureInformation(self) ->Dict:
        return {
            "input_size": self.inputSize,
            "hidden_size": self.hiddenSize,
            "output_size": self.outputSize,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        

