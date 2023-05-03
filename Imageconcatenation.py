import  os

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

from PIL import Image
for j in range(0,16):
    for i in range(1,2):
        im1 = Image.open('/media/moktari/External/TBIOM/Cross_Resolution_Cross_Spectral_GAN/CrossResCrossDomain_Moktari/output/HNIR_Fake_LNIR/%d_1_fake.png'%j)
        im2 = Image.open('/media/moktari/External/TBIOM/Cross_Resolution_Cross_Spectral_GAN/CrossResCrossDomain_Moktari/output/HVIS_real_sample1/%d_1_real.png' %j)
        get_concat_h_blank(im1, im2, (255, 255, 255)).save('/media/moktari/External/TBIOM/Cross_Resolution_Cross_Spectral_GAN/CrossResCrossDomain_Moktari/output/real_fake_concatenated/%d_1_concatenated.png' %j)

