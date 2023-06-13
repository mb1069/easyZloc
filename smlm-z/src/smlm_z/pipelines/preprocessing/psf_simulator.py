# import jax.numpy as np
# import jax 

# import poppy 

# import jax.numpy as jnp
# import jax
# import numpy as np
# import sys
# sys.path.append('/home/miguel/Projects/uni/phd/smlm_z')
# from data.visualise import grid_psfs, show_psf_axial
# def rms2(ab, pupil):
#     return np.sqrt(np.sum(ab**2) / np.sum(pupil <= 1))

# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# try:
#     import cupy as cp

#     print('cupy imported')
#     import_cp = True
# except:
#     import_cp = False

# import matplotlib.pyplot as plt


# def rms(pupil):
#     return jnp.sqrt(jnp.mean(jnp.abs(pupil)**2))
          
# def norm_zero_one(psf):
#     return (psf - psf.min()) / (psf.max() - psf.min())


# import poppy 

# KR = None
# THETA = None
# class Simulator:
    
    
# # pixel size (nm) - 115
# # step size (nm) - 20nm
# # magnification - 100x
# # illumination NA
# # NA at sample
# # refractive index at sample - PBS so 1.335
# # illumination wavelength - 635nm
# # detection wavelength - to check


# # #     N = 2048  # points to use in FFT
# #     Nn = 32
# #     pixel_size = 0.115  # camera pixel size
# #     zrange = 1.0  # distance either side of focus to calculate, in microns, could be arbitrary
# #     dz = 0.02  # step size in axial direction of PSF
    
# #     magnification = 100  # objective magnification
# #     ill_NA = 1.4  # numerical aperture at illumination beams
# #     det_NA = 1.1  # numerical aperture at sample
# #     n = 1.335  # refractive index at sample
# #     fwhmz = 3.0  # FWHM of light sheet in z

# #     ill_wavelength = 635  # illumination wavelength in nm
# #     det_wavelength = 635  # detection wavelength in nm

#     def __init__(self, Nn=32, n_coefs=32, pixel_size=0.90, zrange=1.0, dz=0.02, magnification=100, ill_NA=1.4, det_NA=1.1, n=1.335, fwhmz=3.0, ill_wavelength=635, det_wavelength=635):
#         self.Nn = Nn # lateral size of output PSF
#         self.pixel_size = pixel_size
#         self.zrange = zrange
#         self.dz = dz # step size in axial direction of PSF
#         self.magnification = magnification
#         self.ill_NA = ill_NA # numerical aperture at illumination beams
#         self.ill_wavelength = ill_wavelength # illumination wavelength in nm
#         self.det_NA = det_NA # numerical aperture at sample
#         self.det_wavelength = det_wavelength # detection wavelength in nm
        
#         self.n = n # refractive index at sample
#         self.fwhmz = fwhmz
        
        
# #         np.random.seed(RANDOM_SEED)
#         # self.seed(1234)  # set random number generator seed
#         self.ill_wavelength = self.ill_wavelength * 1e-3
#         self.det_wavelength = self.det_wavelength * 1e-3
#         self.sigmaz = self.fwhmz / 2.355
#         self.dx = self.pixel_size / self.magnification  # Sampling in lateral plane at the sample in um
#         self.dxn = self.det_wavelength / (4 * self.det_NA)  # 2 * Nyquist frequency in x and y.
# #         self.Nn = int(np.ceil(self.N * self.dx / self.dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
        
#         self.N = self.Nn * (self.det_wavelength / (4 * self.det_NA)) / (self.pixel_size / self.magnification)

#         self.dxn = self.N * self.dx / self.Nn  # correct spacing
#         self.res = self.det_wavelength / (2 * self.det_NA)
#         oversampling = self.res / self.dxn  # factor by which pupil plane oversamples the coherent psf data
#         self.dk = oversampling / (self.Nn / 2)  # Pupil plane sampling
#         self.k0 = 2 * jnp.pi * self.n / self.det_wavelength
#         self.kx, self.ky = jnp.meshgrid(jnp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn),
#                                        jnp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn))
#         self.kr = jnp.sqrt(self.kx ** 2 + self.ky ** 2)  # Raw pupil function, pupil defined over circle of radius 1.
        
#         self.krmax = self.det_NA * self.k0 / self.n
#         self.kr2 = self.kx ** 2 + self.ky ** 2
#         self.csum = sum(sum((self.kr < 1)))  # normalise by csum so peak intensity is 1

#         self.alpha = jnp.arcsin(self.det_NA / self.n)
#         # Nyquist sampling in z, reduce by 10 % to account for gaussian light sheet
#         self.dzn = 0.8 * self.det_wavelength / (2 * self.n * (1 - jnp.cos(self.alpha)))
#         self.Nz = int(2 * jnp.ceil(self.zrange / self.dz))
#         self.dz = 2 * self.zrange / self.Nz
#         self.Nzn = int(2 * jnp.ceil(self.zrange / self.dzn))
#         self.dzn = 2 * self.zrange / self.Nzn
#         if self.Nz < self.Nzn:
#             self.Nz = self.Nzn
#             self.dz = self.dzn
#         else:
#             self.Nzn = self.Nz
#             self.dzn = self.dz
            
#         self.n_coefs = n_coefs
#         self.zerns = jnp.zeros((n_coefs, Nn, Nn))

#         theta = jnp.arctan2(self.ky, self.kx)
        
#         masked_pupil = np.array(self.kr)
#         masked_pupil[masked_pupil > 1] = 0
#         masked_pupil = norm_zero_one(masked_pupil)
        
#         KR = np.array(self.kr)
#         THETA = np.array(theta)
# #         plt.imshow(KR)
# #         plt.colorbar()
# #         plt.show()
# #         plt.imshow(THETA)
# #         plt.colorbar()
# #         plt.show()
        
#         for i in range(n_coefs):
#             self.zerns = self.zerns.at[i, :, :].set(poppy.zernike.zernike1(i+1, outside=0.0, rho=masked_pupil, theta=np.array(theta)))
            
#             self.zerns = self.zerns.at[i].set(self.zerns[i] * (self.kr<1).astype(int))
# #             print(rms2(self.zerns[i], self.kr))
        
#     def get_scalar_psf(self, offset=0, zern_coefs=None):
#         if zern_coefs is not None:
#             extra_aberration = jnp.sum(self.zerns * zern_coefs[:, jnp.newaxis, jnp.newaxis], axis=0)
#         else:
#             extra_aberration = 0

#         pupil = self.kr < 1
        
# #         rms_val = rms2(extra_aberration, pupil)
# #         print(np.median(extra_aberration))
# #         plt.title(f'RMS: {rms_val:.2E} min: {extra_aberration.min():.2E} max: {extra_aberration.max():.2E}')
        
# #         plt.imshow(extra_aberration)
# #         plt.show()
        
        
# #         pupil = pupil *  np.exp(1j* aberrations)
# #         nz = 0
# #         psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
# #         for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
# #             c = (np.exp(
# #                 1j * (z * self.n * 2 * np.pi / self.det_wavelength *
# #                       np.sqrt(1 - (self.kr * pupil) ** 2 * self.det_NA ** 2 / self.n ** 2)))) * pupil
# #             psf.at[nz, :, :].set(abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2))
# #             nz = nz + 1
    

# #     With aberrations
# #         nz = 0
# #         psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))
# #         for z in np.arange(-self.zrange, self.zrange - self.dzn, self.dzn):
# #             c = (np.exp(1j * (extra_aberration + z * self.n * 2 * np.pi / self.det_wavelength *
# #                       np.sqrt(1 - (self.kr * pupil) ** 2 * self.det_NA ** 2 / self.n ** 2)))) * pupil
# #             psf = psf.at[nz, :, :].set(abs(np.fft.fftshift(np.fft.ifft2(c))) ** 2 * np.exp(-z ** 2 / 2 / self.sigmaz ** 2))
# #             nz = nz + 1

#         # Optimised version
#         mult1 = self.n * 2 * jnp.pi / self.det_wavelength * jnp.sqrt(1 - (self.kr * pupil) ** 2 * self.det_NA ** 2 / self.n ** 2)
#         zs = jnp.arange(-self.zrange, self.zrange, self.dzn) + offset
#         cs = jax.lax.map(lambda z: (jnp.exp(1j * (extra_aberration + (z * mult1)))) * pupil, zs)
#         cs_psf = abs(jnp.fft.fftshift(jnp.fft.ifft2(cs, axes=(1,2)), axes=(1,2))) ** 2
#         psf = jnp.multiply(cs_psf, jnp.exp(-zs ** 2 / 2 / self.sigmaz ** 2)[:, jnp.newaxis, jnp.newaxis])
                                
#         # Normalised so power in resampled psf(see later on) is unity in focal plane
#         psf = psf * self.Nn ** 2 / jnp.sum(pupil) * self.Nz / self.Nzn

#         return psf

    

# import os

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.01'

# import jax.numpy as jnp
# import jax
# import numpy as np

# def rms2(ab, pupil):
#     return np.sqrt(np.sum(ab**2) / np.sum(pupil <= 1))


# def rms(pupil):
#     return jnp.sqrt(jnp.mean(jnp.abs(pupil)**2))
          
# def norm_zero_one(psf):
#     return (psf - psf.min()) / (psf.max() - psf.min())



# optical_params = dict(Nn=31, 
#                           pixel_size=0.115, 
#                           zrange=1.5, 
#                           dz=0.02, 
#                           magnification=111.11,
#                           ill_NA=1.4,
#                           det_NA=1.3,
#                           n=1.335,
#                           ill_wavelength=635,
#                           det_wavelength=635)

# s = Simulator(**optical_params)


# from jax.numpy.fft import fftn, fftshift, ifftn, ifftshift
# import matplotlib.pyplot as plt
# import sys
# sys.path.append('/home/miguel/Projects/uni/phd/smlm_z')
# from data.visualise import show_psf_axial

# def mask_psf(img, otf_mask=s.kr<=2, alpha=0.1, beta=1, plot=False):
    
#     fft = fftshift(fftn(img, axes=(1,2)), axes=(1,2))
    

#     att_mask = (1 - beta * jnp.exp(-s.kr ** 2 / (2 * alpha ** 2)))

    

#     mask = att_mask * (otf_mask).astype(int)
    
#     if plot:
#         show_psf_axial(norm_zero_one(np.real(fft)), 'imag')
#         show_psf_axial(norm_zero_one(np.imag(fft)), 'real')        
#         plt.imshow(mask)
#         plt.show()
        
#     fft = fft.at[:].set(jnp.multiply(fft[:], mask))
    
#     if plot:
#         show_psf_axial(norm_zero_one(np.real(fft)), 'imag_masked')
#         show_psf_axial(norm_zero_one(np.imag(fft)), 'real_masked')
#         plt.imshow(att_mask)
#         plt.colorbar()
#         plt.show()
#         plt.imshow(np.imag(fft[50]))
#         plt.show()
#         show_psf_axial(np.imag(fft), 'Masked FFT')
    
#     img2 = np.abs(ifftn(ifftshift(fft, axes=(1,2)), axes=(1,2)))
    
#     if plot:
#         show_psf_axial(img, 'Original')
#         show_psf_axial(img2, 'Denoised')
        
#     img2[img2<0] = 0
#     return img2

# # original_psf = s.get_scalar_psf()
# # show_psf_axial(psf)

# # import numpy as np
# # psf = original_psf + np.random.normal(0, 0.01, size=psf.shape)
# # show_psf_axial(psf)

# # for alpha in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
# #     masked_psf = mask_psf(psf.copy(), alpha=alpha)
# #     show_psf_axial(masked_psf)

# #     diff = original_psf-masked_psf
# #     show_psf_axial(diff)
    
# #     print(alpha, diff.min(), diff.max(), abs(diff).sum())
