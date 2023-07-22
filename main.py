import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_ubyte


def generate_gaussian_noise(shape=(1,), mean=0, std_dev=1):
    size = np.prod(shape)
    u1 = np.random.random(size)
    u2 = np.random.random(size)

    z1 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z2 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)

    noise = (z1 * std_dev) + mean
    noise = noise.reshape(shape)
    return noise


def svd_denoising(image, singular_values_fraction):
    image = img_as_float(image)

    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    denoised_red_channel = svd_channel_denoising(red_channel, singular_values_fraction)
    denoised_green_channel = svd_channel_denoising(green_channel, singular_values_fraction)
    denoised_blue_channel = svd_channel_denoising(blue_channel, singular_values_fraction)

    denoised_red_channel = np.clip(denoised_red_channel, -1, 1)
    denoised_green_channel = np.clip(denoised_green_channel, -1, 1)
    denoised_blue_channel = np.clip(denoised_blue_channel, -1, 1)

    denoised_image = np.dstack((denoised_red_channel, denoised_green_channel, denoised_blue_channel))

    return denoised_image


def svd_channel_denoising(channel, singular_values_fraction):
    U, S, V = np.linalg.svd(channel, full_matrices=False)

    num_singular_values = int(singular_values_fraction * len(S))
    S[num_singular_values:] = 0

    denoised_channel = np.dot(U, np.dot(np.diag(S), V))

    return denoised_channel


def main():
    image_directory = './image'

    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_directory, filename)
            image1 = Image.open(image_path)
            image_array = np.array(image1)

            image_array_float = image_array.astype(float)

            mean = random.randrange(-100, 0)
            print(mean)
            std_dev = 30
            shape = image_array_float.shape
            noise = generate_gaussian_noise(shape, mean, std_dev)
            noisy_image = np.clip(image_array_float + noise, 0, 255).astype(np.uint8)
            k = 0.5
            denoised_image = svd_denoising(noisy_image, k)
            denoised_image = img_as_ubyte(denoised_image)

            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes[0].imshow(image1)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(noisy_image.astype(np.uint8))
            axes[1].set_title('Noisy Image')
            axes[1].axis('off')
            axes[2].imshow(denoised_image.astype(np.uint8))
            axes[2].set_title('Denoised Image')
            axes[2].axis('off')
            plt.tight_layout()
            plt.show()
            image1.close()


if __name__ == '__main__':
    main()
