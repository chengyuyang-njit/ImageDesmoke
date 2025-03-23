import pytorch_ssim

def ssim_loss(pred, target):
    return 1 - pytorch_ssim.ssim(pred, target)