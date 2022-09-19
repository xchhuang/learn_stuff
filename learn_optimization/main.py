import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from customized_adam import CustomizedAdam


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_init = np.random.rand(1, 400, 2)
    x1 = torch.from_numpy(x_init).float().to(device)
    x2 = torch.from_numpy(x_init).float().to(device)
    
    target = torch.rand(1, 400, 2).float().to(device).detach()

    loss_adam = []
    loss_customized_adam = []
    adam_opt = torch.optim.Adam([x1.requires_grad_()], lr=1e-3)
    customized_adam_opt = CustomizedAdam([x2.requires_grad_()], device=device, lr=1e-1, n=400, no_dims=2)
    epochs = 100

    for epoch in range(epochs):
        adam_opt.zero_grad()
        out = 2 * x1
        loss = F.mse_loss(out, target)
        loss.backward()
        adam_opt.step()
        loss_adam.append(loss.item())
    
    for epoch in range(epochs):
        customized_adam_opt.zero_grad()
        out = 2 * x2
        loss = F.mse_loss(out, target)
        loss.backward()
        customized_adam_opt.step()
        loss_customized_adam.append(loss.item())

    plt.figure(1)
    plt.plot(loss_adam, 'b-')
    plt.plot(loss_customized_adam, 'g--')
    plt.show()


    
if __name__ == '__main__':
    main()