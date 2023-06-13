import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.01'

import jax.numpy as jnp
import jax
import numpy as np

def rms2(ab, pupil):
    return np.sqrt(np.sum(ab**2) / np.sum(pupil <= 1))


def rms(pupil):
    return jnp.sqrt(jnp.mean(jnp.abs(pupil)**2))
          
def norm_zero_one(psf):
    return (psf - psf.min()) / (psf.max() - psf.min())

import poppy 

KR = None
THETA = None
class Simulator:
    
    def __init__(self, Nn=32, n_coefs=32, pixel_size=0.90, zrange=1.0, dz=0.02, magnification=100, ill_NA=1.4, det_NA=1.1, n=1.335, ill_wavelength=635, det_wavelength=635):
        self.Nn = Nn # lateral size of output PSF
        self.pixel_size = pixel_size
        self.zrange = zrange
        self.dz = dz # step size in axial direction of PSF
        self.magnification = magnification
        self.ill_NA = ill_NA # numerical aperture at illumination beams
        self.ill_wavelength = ill_wavelength # illumination wavelength in nm
        self.det_NA = det_NA # numerical aperture at sample
        self.det_wavelength = det_wavelength # detection wavelength in nm
        
        self.n = n # refractive index at sample
#         self.fwhmz = fwhmz
        
        
#         np.random.seed(RANDOM_SEED)
        # self.seed(1234)  # set random number generator seed
        self.ill_wavelength = self.ill_wavelength * 1e-3
        self.det_wavelength = self.det_wavelength * 1e-3
#         self.sigmaz = self.fwhmz / 2.355
        self.dx = self.pixel_size / self.magnification  # Sampling in lateral plane at the sample in um
        self.dxn = self.det_wavelength / (4 * self.det_NA)  # 2 * Nyquist frequency in x and y.
#         self.Nn = int(np.ceil(self.N * self.dx / self.dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
        
        self.N = self.Nn * (self.det_wavelength / (4 * self.det_NA)) / (self.pixel_size / self.magnification)

        self.dxn = self.N * self.dx / self.Nn  # correct spacing
        self.res = self.det_wavelength / (2 * self.det_NA)
        oversampling = self.res / self.dxn  # factor by which pupil plane oversamples the coherent psf data
        self.dk = oversampling / (self.Nn / 2)  # Pupil plane sampling
        self.k0 = 2 * jnp.pi * self.n / self.det_wavelength
        self.kx, self.ky = jnp.meshgrid(jnp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn),
                                       jnp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn))
        self.kr = jnp.sqrt(self.kx ** 2 + self.ky ** 2)  # Raw pupil function, pupil defined over circle of radius 1.
        
        self.krmax = self.det_NA * self.k0 / self.n
        self.kr2 = self.kx ** 2 + self.ky ** 2
        self.csum = sum(sum((self.kr < 1)))  # normalise by csum so peak intensity is 1

        self.alpha = jnp.arcsin(self.det_NA / self.n)
        # Nyquist sampling in z, reduce by 10 % to account for gaussian light sheet
        self.dzn = 0.8 * self.det_wavelength / (2 * self.n * (1 - jnp.cos(self.alpha)))
        self.Nz = int(2 * jnp.ceil(self.zrange / self.dz))
        self.dz = 2 * self.zrange / self.Nz
        self.Nzn = int(2 * jnp.ceil(self.zrange / self.dzn))
        self.dzn = 2 * self.zrange / self.Nzn
        if self.Nz < self.Nzn:
            self.Nz = self.Nzn
            self.dz = self.dzn
        else:
            self.Nzn = self.Nz
            self.dzn = self.dz
            
        self.n_coefs = n_coefs
        self.zerns = jnp.zeros((n_coefs, Nn, Nn))

        theta = jnp.arctan2(self.ky, self.kx)
        
        masked_pupil = np.array(self.kr)
        masked_pupil[masked_pupil > 1] = 0
        masked_pupil = norm_zero_one(masked_pupil)
        
        KR = np.array(self.kr)
        THETA = np.array(theta)
#         plt.imshow(KR)
#         plt.colorbar()
#         plt.show()
#         plt.imshow(THETA)
#         plt.colorbar()
#         plt.show()
        
        for i in range(n_coefs):
            self.zerns = self.zerns.at[i, :, :].set(poppy.zernike.zernike1(i+1, outside=0.0, rho=masked_pupil, theta=np.array(theta)))
            
            self.zerns = self.zerns.at[i].set(self.zerns[i] * (self.kr<1).astype(int))
            
#       Remove defocus
        self.zerns = self.zerns.at[3].set(0)
#             print(rms2(self.zerns[i], self.kr))
        
    def get_scalar_psf(self, offset=0, zern_coefs=None):

#         extra_aberration = jnp.zeros(self.kr.shape)
#         aberrations = jnp.array([
#             jnp.ones(self.kr.shape), # piston
#             2 * self.kx, # tipx
#             2 * self.ky, # tilt
#             jnp.sqrt(3) * ((2 * (self.kr**2)) - 1), # defocus
#             2 * jnp.sqrt(6) * (self.kx * self.ky), # vertical astigmatism
#             jnp.sqrt(6) * (self.kx**2 - self.ky ** 2), # oblique astigmatism
#             2 * jnp.sqrt(2) * (3 * self.kr**2 - 2) * self.ky, # horizontal coma
#             2 * jnp.sqrt(2) * (3 * self.kr**2 - 2) * self.kx, # vertical coma
#             2 * jnp.sqrt(2) * (3 * self.kr**2 - 4 * self.kr**2) * self.ky, # vertical trefoil 
#             2 * jnp.sqrt(2) * (4 * self.kx**2 - 3 * self.kr**2) * self.kx, # oblique trefoil 
#             jnp.sqrt(5) * (6 * self.kr**2 * (self.kr**2 - 1) + 1), # primary spherical
#             jnp.sqrt(10) * (8 * self.kx**4 - 8 * self.kx**2 * self.ky**2 - (3 * self.kx ** 2)), # vertical secondary astigmatism
#             jnp.sqrt(10) * (4 * self.kx * self.ky) * (4 * self.kx**2 + 4 * self.ky**2 - 3), # oblique secondary astigmatism
#             jnp.sqrt(10) * ((self.kx**2 - self.ky**2) * (4*self.kx**2 - 3)) ** 2, # vertical quadrafoil
#             jnp.sqrt(10) * ((2 * self.kx * self.ky) * (self.kx**2 - self.ky**2)) ** 2 # oblique quadrafoil
#         ])
    

        
        
#         for custom, p in zip(aberrations, self.zerns):
#             import numpy as np
# #             custom = custom.at[~(self.kr<1)].set(0)
#             custom = norm_zero_one(custom.copy())
#             p = norm_zero_one(p.copy())
#             plt.imshow(np.concatenate((custom, p), axis=1))
#             plt.show()
#             print(custom.min(), custom.max())
#             print(p.min(), p.max())
#         return
        

    
#         if zern_coefs is not None:
#             n_coefs = zern_coefs.shape[0]
#             aberrations = aberrations[:n_coefs]
#             cust_extra_aberration = jnp.multiply(aberrations, zern_coefs[:, jnp.newaxis, jnp.newaxis]).sum(axis=0)
#         else:
#             cust_extra_aberration = 0

        if zern_coefs is not None:
            extra_aberration = jnp.sum(self.zerns * zern_coefs[:, jnp.newaxis, jnp.newaxis], axis=0)
        else:
            extra_aberration = 0

        pupil = self.kr < 1
#         rms_val = rms2(extra_aberration, pupil)
#         print(np.median(extra_aberration))
#         plt.title(f'RMS: {rms_val:.2E} min: {extra_aberration.min():.2E} max: {extra_aberration.max():.2E}')
        
#         plt.imshow(extra_aberration)
#         plt.show()
        
        
#         pupil = pupil *  np.exp(1j* aberrations)
#         nz = 0
#         psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
#         for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
#             c = (np.exp(
#                 1j * (z * self.n * 2 * np.pi / self.det_wavelength *
#                       np.sqrt(1 - (self.kr * pupil) ** 2 * self.det_NA ** 2 / self.n ** 2)))) * pupil
#             psf.at[nz, :, :].set(abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2))
#             nz = nz + 1
    

#     With aberrations
#         nz = 0
#         psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
#         for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
#             c = (np.exp(1j * (extra_aberration + z * self.n * 2 * np.pi / self.det_wavelength *
#                       np.sqrt(1 - (self.kr * pupil) ** 2 * self.det_NA ** 2 / self.n ** 2)))) * pupil
#             psf = psf.at[nz, :, :].set(abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2))
#             nz = nz + 1

        # Optimised version
        mult1 = self.n * 2 * jnp.pi / self.det_wavelength * jnp.sqrt(1 - (self.kr * pupil) ** 2 * self.det_NA ** 2 / self.n ** 2)
        zs = jnp.arange(-self.zrange, self.zrange, self.dzn) + offset
        cs = jax.lax.map(lambda z: (jnp.exp(1j * (extra_aberration + (z * mult1)))) * pupil, zs)
        psf = abs(jnp.fft.fftshift(jnp.fft.ifft2(cs, axes=(1,2)), axes=(1,2))) ** 2
        # Normalised so power in resampled psf(see later on) is unity in focal plane
        psf = psf * self.Nn ** 2 / jnp.sum(pupil) * self.Nz / self.Nzn

        return psf



from jax.numpy.fft import fftn, fftshift, ifftn, ifftshift
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/miguel/Projects/uni/phd/smlm_z')
from data.visualise import show_psf_axial

def mask_psf(img, simul_kr, otf_mask, alpha=0.01, beta=1, plot=False):
    
    fft = fftshift(fftn(img, axes=(1,2)), axes=(1,2))
    

    att_mask = (1 - beta * jnp.exp(-simul_kr ** 2 / (2 * alpha ** 2)))

    

    mask = att_mask * (otf_mask).astype(int)
    
    if plot:
        plt.rcParams["figure.figsize"] = (10, 5)
        show_psf_axial(norm_zero_one(np.real(fft)), 'imag')
        show_psf_axial(norm_zero_one(np.imag(fft)), 'real')        
        plt.imshow(mask)
        plt.show()
        
    fft = fft.at[:].set(jnp.multiply(fft[:], mask))
    
    if plot:
        show_psf_axial(norm_zero_one(np.real(fft)), 'imag_masked')
        show_psf_axial(norm_zero_one(np.imag(fft)), 'real_masked')
        plt.imshow(att_mask)
        plt.colorbar()
        plt.show()
        plt.imshow(np.imag(fft[50]))
        plt.show()
        show_psf_axial(np.imag(fft), 'Masked FFT')
    
    img2 = np.abs(ifftn(ifftshift(fft, axes=(1,2)), axes=(1,2)))
    
    if plot:
        show_psf_axial(img, 'Original')
        show_psf_axial(img2, 'Denoised')
        
    img2[img2<0] = 0
    return img2

def mask_psfs(psfs, parameters):
    
    optical_params = dict(Nn=parameters['images']['picasso_box'], 
                          pixel_size=parameters['images']['xy_res']/1000, 
                          zrange=parameters['z_range']/1000, 
                          dz=parameters['images']['train']['z_step']/1000, 
                          magnification=parameters['images']['magnification'],
                          ill_NA=parameters['images']['ill_NA'],
                          det_NA=parameters['images']['det_NA'],
                          n=parameters['images']['n'],
                          ill_wavelength=parameters['images']['ill_wavelength'],
                          det_wavelength=parameters['images']['det_wavelength'])
    simul = Simulator(**optical_params)
    otf_mask = simul.kr<=2

    return np.concatenate([mask_psf(psf[np.newaxis], simul_kr=simul.kr, otf_mask=otf_mask) for psf in psfs])