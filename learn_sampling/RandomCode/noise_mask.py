import numpy as np
import matplotlib.pyplot as plt


def main():
    res = 64
    mask = np.ones((res, res)).astype(np.float32)
    mask[res//4:res//4 * 3, res//4:res//4 * 3] = 0

    white_fft_avg = 0
    white_filtered_fft_avg = 0
    
    num_realizations = 1000
    for _ in range(num_realizations):
        white_noise = np.random.randn(res, res)

        # mu = np.mean(white_noise)
        # white_noise_back = np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(white_noise - mu))) + mu).real
        # print(white_noise.shape, white_noise_back.shape)
        # err = np.sum((white_noise - white_noise_back) ** 2)
        # print('err:', err)
        
        # white_fft = np.abs(np.fft.fftshift(np.fft.fft2(white_noise - np.mean(white_noise))/ white_noise.shape[0]))
        white_fft = np.abs(np.fft.fftshift(np.fft.fft2(white_noise)))
        
        white_filtered = np.fft.ifft2(white_fft).real
        white_filtered_fft = np.abs(np.fft.fftshift(np.fft.fft2(white_filtered - np.mean(white_filtered))/ white_filtered.shape[0]))

        # white_fft = white_fft * mask
        white_fft_avg += white_fft / num_realizations
        white_filtered_fft_avg += white_filtered_fft / num_realizations
        
    plt.figure(1)
    plt.subplot(121)
    plt.title('White noise')
    plt.imshow(white_filtered_fft_avg)
    plt.axis('off')
    plt.subplot(122)
    plt.title('Blue noise')
    plt.imshow(white_fft_avg)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()

