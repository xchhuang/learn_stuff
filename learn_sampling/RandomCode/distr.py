import numpy as np  
import matplotlib.pyplot as plt
import torch
from scipy.stats import gennorm, norm


# uniform distribution range
a = -5
b = 5


def sample_normal(n, mean, std_dev):
    # samples = np.random.normal(mean, std_dev, n)
    samples = np.linspace(a, b, n)
    pdf_values = norm.pdf(samples, loc=mean, scale=std_dev)
    return samples, pdf_values


def sample_z(n, mu, alpha, beta):
    # Create a generalized normal distribution object
    dist = gennorm(beta, loc=mu, scale=alpha)
    # samples = dist.rvs(size=n)
    samples = np.linspace(a, b, n)
    pdf_values = dist.pdf(samples)
    return samples, pdf_values



def main():
    n = 100000
    z, p = sample_z(n=100000, mu=0.0, alpha=1.0, beta=8.0)
    # z, p = sample_normal(n=n, mean=0.0, std_dev=1.0)
    print('pdf sum:', z, p, np.mean(p)*(b-a))
    plt.figure(1)
    # plt.title(str(np.mean(p)))
    # plt.hist(z, bins=100)
    plt.plot(np.linspace(a, b, n), p)
    plt.show()


if __name__ == '__main__':
    main()

