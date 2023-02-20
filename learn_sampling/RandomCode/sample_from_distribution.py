import torch
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

beta_start = 0.0001
beta_end = 0.02
T = 1000
theta_0 = 0.001
betas = torch.linspace(beta_start, beta_end, T).to(device)


def sample_from_gamma():
    x = torch.randn(1).to(device)
    b = betas
    a = (1 - b).cumprod(dim=0)
    k = (b / a) / theta_0 ** 2
    theta = (a.sqrt() * theta_0)
    k_bar = k.cumsum(dim=0)
    concentration = torch.ones(x.size()).to(device) * k_bar[-1]
    rates = torch.ones(concentration.size()).to(device) * theta[-1]
    m = torch.distributions.Gamma(concentration, 1 / rates)
    x = m.sample() - rates * concentration
    # print(x)
    # pdf = torch.exp(m.log_prob(x))
    return x
    

def sample_from_gaussian():
    x = torch.randn(1).to(device)
    x = torch.randn_like(x)
    return x


def sample_from_uniform():
    # x = torch.rand(1).to(device) - 0.5
    m = torch.distributions.Uniform(torch.tensor([-np.sqrt(3)]).to(device), torch.tensor([np.sqrt(3)]).to(device))
    x = m.sample()
    pdf = m.log_prob(x).exp()
    
    return x, pdf


def generalized_uniform():
    x = torch.randn(1).to(device)
    with torch.no_grad():
        
        b = betas
        a = (1 - b).cumprod(dim=0)
        k = (b / a)/theta_0**2
        theta = (a.sqrt()*theta_0)
        k_bar = k.cumsum(dim=0)
        seq = range(0, T, 1)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):
            eps_s1 = []
            eps_s2 = []
            print(j)
            if j == 2:
                for k in tqdm(range(100000)):
                    concentration = torch.ones(x.size()).to(x.device) * k_bar[j]
                    rates = torch.ones(x.size()).to(x.device) * theta[j]
                    m = torch.distributions.Gamma(concentration, 1 / rates)
                    eps = m.sample()
                    eps = eps - concentration * rates
                    
                    # print('eps:', i, j, eps.shape, eps[0].min(), eps[0].max(), eps[0].mean(), eps[0].std())
                    # eps = torch.randn_like(eps)
                    eps_s1.append(eps.detach().cpu().numpy())

                    eps = eps / (1.0 - a[j]).sqrt()

                    eps_s2.append(eps.detach().cpu().numpy())
                    

                eps_s1 = np.concatenate(eps_s1, 0)
                eps_s2 = np.concatenate(eps_s2, 0)
                
                print(eps_s1.shape, np.mean(eps_s1), np.var(eps_s1))
                print(eps_s2.shape, np.mean(eps_s2), np.var(eps_s2))
                
                return

            # eps = torch.randn_like(eps)
            # eps = (torch.rand_like(eps) - 0.5) * np.sqrt(12)

            # print('eps:', i, j, eps.shape, eps[0].min(), eps[0].max(), eps[0].mean(), eps[0].std())




def uniform_estimation_loss():
    
    x0 = torch.randn(1).to(device)
    t = 2
    
    
    e_s1 = []
    e_s2 = []
    
    for i in tqdm(range(100000)):
        b = betas
        a = (1 - b).cumprod(dim=0)
        # print('tba:', t, b[t])
        k = (b / a)/theta_0**2
        
        theta = (a.sqrt()*theta_0)[t]
        k_bar = k.cumsum(dim=0)[t]
        # print('k_bar:', k_bar, k)
        a = a[t]
        # b_t = b.index_select(0, t).view(-1, 1, 1, 1)
        concentration = torch.ones(x0.size()).to(x0.device) * k_bar
        rates = torch.ones(x0.size()).to(x0.device) * theta
        m = torch.distributions.Gamma(concentration, 1 / rates)
        e = m.sample()
        # print('uniform_estimation_loss e1:', e.shape, e.min(), e.max(), e.mean(), e.std()) 
        e = e - concentration * rates

        e_s1.append(e.detach().cpu().numpy())
        
        e = e / (1.0 - a).sqrt()

        e_s2.append(e.detach().cpu().numpy())
    
    e_s1 = np.concatenate(e_s1, 0)
    e_s2 = np.concatenate(e_s2, 0)
    print(e_s1.shape, np.mean(e_s1), np.var(e_s1))
    print(e_s2.shape, np.mean(e_s2), np.var(e_s2))
    


def main():

    # sample from a pdf
    # xs = []
    # for i in range(10000):
    #     # x = sample_from_gamma()
    #     # x = sample_from_gaussian()
    #     x, pdf = sample_from_uniform()
    #     xs.append(x.detach().cpu().numpy())
    # xs = np.concatenate(xs)
    # print(xs.shape)
    # print(np.mean(xs), np.var(xs))

    # sample sequence
    # generalized_uniform()

    # sample at t
    uniform_estimation_loss()

if __name__ == '__main__':
    main()
