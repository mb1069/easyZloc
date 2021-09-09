import numpy as np
from tifffile import imread
from final_project.smlm_3d.data.visualise import show_psf_axial
# Noise simulation
img = imread('/home/miguel/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters/91.tif')

img = img / img.max()

def generate_noisy_psf(img):
    img = img / img.max()

    background_noise_percentage = np.random.uniform(0.75, 0.8)

    pixel_val_range = img.max() - img.min()
    mean_scale = pixel_val_range/100
    scale = np.random.uniform(mean_scale * 0.6, mean_scale * 1.4)
    noise = np.random.normal(loc=background_noise_percentage, scale=scale, size=img.shape)
    noise_img = img + noise

    noise_img[noise_img <= 0] = 0

    noise_img = noise_img * 255

    noise_img = np.random.poisson(noise_img.astype(int), size=img.shape)

    noise_img = noise_img / noise_img.max()

    return noise_img

if __name__=='__main__':
    noise_imgs = []
    for _ in range(10):
        
        noise_img = generate_noisy_psf(img, snr)
        noise_imgs.append(noise_img)

    combined = np.concatenate((img, *noise_imgs), axis=2)

    show_psf_axial(combined, subsample_n=3)