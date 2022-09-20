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
    customized_adam_opt = CustomizedAdam([x2.requires_grad_()], device=device, lr=1e-3, n=400, no_dims=2)
    epochs = 100

    constant_weight = 3
    for epoch in range(epochs):
        adam_opt.zero_grad()
        out = constant_weight * x1
        loss = F.mse_loss(out, target)
        loss.backward()
        adam_opt.step()
        loss_adam.append(loss.item())
    
    for epoch in range(epochs):
        customized_adam_opt.zero_grad()
        out = constant_weight * x2
        loss = F.mse_loss(out, target)
        loss.backward()
        customized_adam_opt.step()
        loss_customized_adam.append(loss.item())

    # print('losses:', loss_adam, loss_customized_adam)

    err = np.mean((np.array(loss_adam) - np.array(loss_customized_adam)) ** 2)
    plt.figure(1)
    plt.title('error: {:.4f}'.format(err))
    plt.plot(loss_adam, 'b-')
    plt.plot(loss_customized_adam, 'g--')
    plt.show()


    
if __name__ == '__main__':
    main()