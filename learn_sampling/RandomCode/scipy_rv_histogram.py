import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def main():
    # data = scipy.stats.norm.rvs(size=100000, loc=0, scale=1.5, random_state=123)
    data = scipy.stats.uniform.rvs(size=100000, random_state=123)
    hist = np.histogram(data, bins=100)
    hist_dist = scipy.stats.rv_histogram(hist)
    #
    X = np.linspace(0, 1, 100)
    plt.title("PDF from Template")
    plt.hist(data, density=True, bins=100)
    plt.plot(X, hist_dist.pdf(X), label='PDF')
    plt.plot(X, hist_dist.cdf(X), label='CDF')
    plt.show()

    samples = hist_dist.rvs(size=100)
    pdfs = hist_dist.pdf(0.1)
    # print(samples, pdfs)


if __name__ == '__main__':
    main()