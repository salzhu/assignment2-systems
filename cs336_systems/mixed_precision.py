import torch 
import torch.nn as nn 

import argparse

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        print('after fc1', end=' ')
        print(x.dtype, self.fc1.weight.dtype)
        x = self.relu(x)
        print('after relu', end=' ')
        print(x.dtype)
        x = self.ln(x)
        print('after ln', end=' ')
        print(x.dtype, self.ln.weight.dtype)
        x = self.fc2(x)
        print('after fc2', end=' ')
        print(x.dtype, self.fc2.weight.dtype)
        return x
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--bfloat16", type=bool, default=False)
    args = parser.parse_args()

    model = ToyModel(10, 10)
    model.to('cuda')

    x = torch.randn(size=(4, 10), device='cuda')

    dtype = torch.float16
    if args.bfloat16 == True: 
        dtype = torch.bfloat16 

    with torch.autocast(device_type='cuda',dtype=dtype):
        print('model weights')
        for name, param in model.named_parameters():
            print(name, param.dtype)

        print('----')

        y = model(x)
        loss = nn.CrossEntropyLoss()
        output = loss(y, torch.randn(size=(4, 10), device='cuda'))
        output.backward()

        print('----')

        print('logits', end = ' ')
        print(y.dtype)

        print('----')

        print('loss', end = ' ')
        print(output.dtype)

        print('----')

        print('gradients')
        for name, param in model.named_parameters():
            print(name, param.grad.dtype)