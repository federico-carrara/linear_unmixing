from math import log10

def PSNR(gt, pred):
    return 10 * (log10(gt.max()) - log10(pixel_wise_mse(gt - pred).mean()))

def pixel_wise_mse(gt, pred):
    return (gt - pred)**2