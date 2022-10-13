from turtle import back
import numpy as np
from tifffile import imread
from data.visualise import show_psf_axial
from data.estimate_offset import norm_zero_one

class EMCCD:
    def __init__(self, noise_background = 0.0, quantum_efficiency=0.9, read_noise=74.4, spurious_charge=0.0002,em_gain=300.0, baseline=100.0, e_per_adu=45.0):
        self.qe = quantum_efficiency
        self.read_noise = read_noise
        self.c = spurious_charge
        self.em_gain = em_gain
        self.baseline = baseline
        self.e_per_adu = e_per_adu
        self.noise_bg = noise_background

    def add_noise(self, photon_counts):
        n_ie = np.random.poisson(self.qe*(photon_counts+self.noise_bg) + self.c)
        n_oe = np.random.gamma(n_ie+0.001, scale=self.em_gain)
        n_oe = n_oe + np.random.normal(0.0,self.read_noise,n_oe.shape)
        ADU_out = (n_oe/self.e_per_adu).astype(int) + self.baseline
        return self.center(np.minimum(ADU_out,65535))

    def gain(n):
        return n.qe*n.em_gain/n.e_per_adu

    def mean(n):
        return (n.noise_bg*n.qe + n.c)*n.em_gain/n.e_per_adu + n.baseline

    def center(self, img):
        return (img - self.mean())/self.gain()


def generate_noisy_psf(img, nsr=None):
    print(img.min(), img.max())
    noise_config = EMCCD(
        noise_background=np.random.uniform(0, 0.4),
        quantum_efficiency=np.random.normal(0.9, 0.1)
    )

    background_noise_loc = nsr or np.random.uniform(0, 0.8)
    background_noise_scale = np.random.uniform(0, 0.1)
    background_noise = np.random.normal(loc=background_noise_loc, scale=background_noise_scale, size=img.shape)
    # background_noise[:] = 0
    noise_img = (img + background_noise) 


    # noise_img[noise_img < 0] = 0

    # noise_img = np.random.poisson(noise_img.astype(int), size=img.shape)
    
    # noise_img = np.clip(noise_img, a_min=0, a_max=1)
    # noise_img = norm_zero_one(noise_img)

    return noise_img

if __name__=='__main__':

    # Noise simulation
    img = imread('/home/miguel/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters/91.tif')

    img = img / img.max()


    noise_imgs = []
    for _ in range(10):
        
        noise_img = generate_noisy_psf(img, snr)
        noise_imgs.append(noise_img)

    combined = np.concatenate((img, *noise_imgs), axis=2)

    show_psf_axial(combined, subsample_n=3)