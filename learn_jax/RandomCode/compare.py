import torch

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')


def func_torch(p):
    out = torch.zeros(1).to(p.device)
    xx = torch.linspace(0, 1, 100).to(device)
    for x in xx:
        # print(x, p)
        if x < p:
            out += 1
        else:
            out += 0.5

    return out


def main():
    inp = torch.rand(1, requires_grad=True).to(device)
    out = func_torch(inp)
    print('(before)grad:', inp.grad)
    out.backward()
    print('(after)grad:', inp.grad)


if __name__ == '__main__':
    main()
