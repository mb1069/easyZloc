from config.datasets import dataset_configs
from data.visualise import show_psf_axial

from scipy import fftpack
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def apply_low_pass(img):
    #fft of image
    fft1 = fftpack.fftshift(fftpack.fft2(img))

    #Create a low pass filter image
    x,y = img.shape
    #size of circle
    e_x,e_y=25,25
    #create a box 
    bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

    low_pass=Image.new("L",(img.shape[0],img.shape[1]),color=0)

    draw1=ImageDraw.Draw(low_pass)
    draw1.ellipse(bbox, fill=1)

    low_pass_np=np.array(low_pass)

    #multiply both the images
    filtered=np.multiply(fft1,low_pass_np)

    #inverse fft
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
    ifft2 = np.maximum(0, np.minimum(ifft2, 255))
    return ifft2

def apply_low_pass_img_stack(imstack):
    img = np.stack(list(map(apply_low_pass, imstack)))
    return img

if __name__=='__main__':
    from data.datasets import TrainingDataSet

    z_range = 1000
    cfg = dataset_configs['openframe']['training']
    dataset = TrainingDataSet(cfg, z_range, lazy=True, transform_data=False)

    dataset.prepare_debug()

    psf = dataset.debug_emitter(1, z_range=1000)[0]

    _img = psf.copy()

    _img_denoised = apply_low_pass_img_stack(_img)

    _img = _img / _img.max()

    _img_denoised = _img_denoised / _img_denoised.max()

    show_psf_axial(np.concatenate((_img, _img_denoised), axis=2))
    plt.show()

    #save the image
