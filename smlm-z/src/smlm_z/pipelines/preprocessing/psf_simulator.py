import jax.numpy as np
import jax 

import poppy 

class Simulator:

    def __init__(self, Nn=32, n_coefs=32, pixel_size=0.90, zrange=1.0, dz=0.02, magnification=100, ill_NA=1.4, det_NA=1.1, n=1.335, fwhmz=3.0, ill_wavelength=635, det_wavelength=635):
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
        self.fwhmz = fwhmz
        
        
#         np.random.seed(RANDOM_SEED)
        # self.seed(1234)  # set random number generator seed
        self.ill_wavelength = self.ill_wavelength * 1e-3
        self.det_wavelength = self.det_wavelength * 1e-3
        self.sigmaz = self.fwhmz / 2.355
        self.dx = self.pixel_size / self.magnification  # Sampling in lateral plane at the sample in um
        self.dxn = self.det_wavelength / (4 * self.det_NA)  # 2 * Nyquist frequency in x and y.
#         self.Nn = int(np.ceil(self.N * self.dx / self.dxn / 2) * 2)  # Number of points at Nyquist sampling, even number
        
        self.N = self.Nn * (self.det_wavelength / (4 * self.det_NA)) / (self.pixel_size / self.magnification)

        self.dxn = self.N * self.dx / self.Nn  # correct spacing
        self.res = self.det_wavelength / (2 * self.det_NA)
        oversampling = self.res / self.dxn  # factor by which pupil plane oversamples the coherent psf data
        self.dk = oversampling / (self.Nn / 2)  # Pupil plane sampling
        self.k0 = 2 * np.pi * self.n / self.det_wavelength
        self.kx, self.ky = self.xp.meshgrid(self.xp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn),
                                       self.xp.linspace(-self.dk * self.Nn / 2, self.dk * self.Nn / 2 - self.dk, self.Nn))
        self.kr = np.sqrt(self.kx ** 2 + self.ky ** 2)  # Raw pupil function, pupil defined over circle of radius 1.
        
        self.krmax = self.det_NA * self.k0 / self.n
        self.kr2 = self.kx ** 2 + self.ky ** 2
        self.csum = sum(sum((self.kr < 1)))  # normalise by csum so peak intensity is 1

        self.alpha = np.arcsin(self.det_NA / self.n)
        # Nyquist sampling in z, reduce by 10 % to account for gaussian light sheet
        self.dzn = 0.8 * self.det_wavelength / (2 * self.n * (1 - np.cos(self.alpha)))
        self.Nz = int(2 * np.ceil(self.zrange / self.dz))
        self.dz = 2 * self.zrange / self.Nz
        self.Nzn = int(2 * np.ceil(self.zrange / self.dzn))
        self.dzn = 2 * self.zrange / self.Nzn
        if self.Nz < self.Nzn:
            self.Nz = self.Nzn
            self.dz = self.dzn
        else:
            self.Nzn = self.Nz
            self.dzn = self.dz
            
        self.n_coefs = n_coefs
        self.zerns = np.zeros((n_coefs, Nn, Nn))

        for i in range(n_coefs):
            self.zerns = self.zerns.at[i, :, :].set(poppy.zernike.zernike1(i+1, npix=Nn, outside=0.0))

    
    def get_scalar_psf(self, zern_coefs=None):
        # use krmax to define the pupil function
        kx = self.krmax * self.kx
        ky = self.krmax * self.ky
        kr2 = (kx ** 2 + ky ** 2)  # square kr
        psf = self.xp.zeros((self.Nzn, self.Nn, self.Nn))

        extra_aberration = np.zeros(self.kr.shape)
        if zern_coefs is not None:
            extra_aberration = np.sum(self.zerns * zern_coefs[:, np.newaxis, np.newaxis], axis=0)
        pupil = self.kr < 1
    
#     Optimised version
        mult1 = self.n * 2 * np.pi / self.det_wavelength * np.sqrt(1 - (self.kr * pupil) ** 2 * self.det_NA ** 2 / self.n ** 2)
        zs = np.arange(-self.zrange, self.zrange, self.dzn)
        cs = jax.lax.map(lambda z: (np.exp(1j * (extra_aberration + (z * mult1)))) * pupil, zs)
        cs_psf = abs(np.fft.fftshift(np.fft.ifft2(cs, axes=(1,2)), axes=(1,2))) ** 2
        psf = np.multiply(cs_psf, np.exp(-zs ** 2 / 2 / self.sigmaz ** 2)[:, np.newaxis, np.newaxis])
                                
        # Normalised so power in resampled psf(see later on) is unity in focal plane
        psf = psf * self.Nn ** 2 / self.xp.sum(pupil) * self.Nz / self.Nzn

        return psf