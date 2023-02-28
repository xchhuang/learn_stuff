import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    res = 64
    mask = np.ones((res, res)).astype(np.float32)
    mask[res//2-3:res//2+3, res//2-3:res//2+3] = 0.0

    white_fft_avg = 0
    white_filtered_fft_avg = 0
    
    num_realizations = 1000
    for _ in range(num_realizations):
        white_noise = np.clip(np.random.randn(res, res), -3, 3)

        mu = np.mean(white_noise)
        
        # print(mu)
        # white_noise_fft = np.fft.fftshift(np.fft.fft2(white_noise - mu))
        # white_noise_back = np.fft.ifft2(np.fft.fftshift(white_noise_fft)).real + mu
        # print(white_noise.shape, white_noise_back.shape, white_noise_back)
        # err = np.sum((white_noise - white_noise_back) ** 2)
        # print('err:', err)
        # return

        # white_fft = np.abs(np.fft.fftshift(np.fft.fft2(white_noise - np.mean(white_noise))/ white_noise.shape[0]))
        white_fft = np.abs(np.fft.fftshift(np.fft.fft2(white_noise - mu)))

        white_fft_filtered = white_fft * mask
        white_filtered = np.fft.ifft2(np.fft.fftshift(white_fft_filtered)).real + mu
        white_filtered = (white_filtered - white_filtered.min()) / (white_filtered.max() - white_filtered.min())
        white_filtered = (white_filtered - 0.5) * 6
        # print(white_filtered.shape, white_filtered.min(), white_filtered.max())
        # print(white_noise.shape, white_noise.min(), white_noise.max())
        
        white_fft_avg += white_fft / num_realizations
        white_filtered_fft_avg += white_fft_filtered / num_realizations
    
    # cv2.imwrite('results/white_filtered_fft_avg.exr', white_filtered_fft_avg.astype('float32'))
    # cv2.imwrite('results/white_fft.exr', white_fft.astype('float32'))
    
    plt.figure(1)
    plt.subplot(221)
    plt.title('White noise')
    plt.imshow(white_noise)
    plt.axis('off')
    plt.subplot(222)
    plt.title('White noise fft')
    plt.imshow(white_fft_avg)
    plt.axis('off')
    plt.subplot(223)
    plt.title('Filtered noise')
    plt.imshow(white_filtered)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(224)
    plt.title('Filtered noise fft')
    plt.imshow(white_filtered_fft_avg)
    plt.show()

if __name__ == '__main__':
    main()

