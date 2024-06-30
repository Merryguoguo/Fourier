# https://github.com/mu-cai/frequency-domain-image-translation/blob/master/utils_freq/freq_pixel_loss.py
def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel

def find_fake_freq(im, gauss_kernel, index=None):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = gaussian_blur(im, gauss_kernel, padding=padding)
    im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
    im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
    low_gray = gaussian_blur(im_gray, gauss_kernel, padding=padding)
    return torch.cat((low_freq, im_gray - low_gray),1)

# https://github.com/mu-cai/frequency-domain-image-translation/blob/master/utils_freq/freq_fourier_loss.py
def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    fft = torch.rfft(image, 2, onesided=False)
    fft_mag = torch.log(1 + torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2 + 1e-8))
    return fft_mag

def decide_circle(N=4,  L=256,r=96, size = 256):
    x=torch.ones((N, L, L))
    for i in range(L):
        for j in range(L):
            if (i- L/2 + 0.5)**2 + (j- L/2 + 0.5)**2 < r **2:
                x[:,i,j]=0
    return x, torch.ones((N, L, L)) - x

def fft_L1_loss_color(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_fft = calc_fft(fake_image)
    real_fft = calc_fft(real_image)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


# https://github.com/mu-cai/frequency-domain-image-translation/blob/master/swapping-autoencoder/train.py
# args.gauss_size: 
# args.batch: 
# args.radius:
parser.add_argument('--gauss_size', type=int, default=21)
parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
parser.add_argument('--radius', type=int, default=21)

def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
    crop_size1 = torch.rand(n_crop) * (max_size - min_size) + min_size
    crop_size2 = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, channel, height, width = img.shape
    if height==256:
        max_size = max_size * 2
    elif height==512 or height==1024:
        # print(height)
        max_size = max_size * 1
    else:
        assert False
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size1 * height).type(torch.int64).tolist()
    crop_w = (crop_size2 * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, height - c_h)
        c_x = random.randrange(0, width - c_w)

        cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

    return patches


from torch.nn import functional as F
def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


from model import Encoder, Generator, Discriminator, CooccurDiscriminator
cooccur = CooccurDiscriminator(args.channel, size=args.size).to(device)

gauss_kernel = get_gaussian_kernel(args.gauss_size).cuda()
mask_h, mask_l =decide_circle(r=args.radius, N=int(args.batch/2), L=args.size)
mask_h, mask_l = mask_h.cuda(), mask_l.cuda()

real_img = next(loader)
real_img = real_img.to(device)
real_img_freq = find_fake_freq(real_img, gauss_kernel)
real_img_freq1, real_img2_freq = real_img_freq.chunk(2, dim=0) 

fake_img1 = generator(structure1, texture1)
fake_img2 = generator(structure1, texture2)
fake_img1_freq = find_fake_freq(fake_img1, gauss_kernel)
fake_img2_freq = find_fake_freq(fake_img2, gauss_kernel)
fake_patch = patchify_image(fake_img2_freq[:, :3, :, :], args.n_crop)
ref_patch = patchify_image(real_img2_freq[:, :3, :, :], args.ref_crop * args.n_crop)
fake_patch_pred, _ = cooccur(fake_patch, ref_patch, ref_batch=args.ref_crop)
g_cooccur_loss = g_nonsaturating_loss(fake_patch_pred)

recon_freq_loss_img1_low = F.l1_loss(fake_img1_freq[:, :3, :, :], real_img_freq1[:, :3, :, :])
recon_freq_loss_img1_high = F.l1_loss(fake_img1_freq[:, 3:6, :, :], real_img_freq1[:, 3:6, :, :])

recon_fft = fft_L1_loss_color(fake_img1, real_img1)
recon_freq_loss_img1 =args.w_low_recon * recon_freq_loss_img1_low + args.w_high_recon * recon_freq_loss_img1_high
recon_freq_loss_img2_structure = F.l1_loss(fake_img2_freq[:, 3:6, :, :], real_img_freq1[:, 3:6, :, :])
fft_swap_H =  fft_L1_loss_mask(fake_img2, real_img1, mask_h)

loss_dict["recon"] = recon_loss
loss_dict["g"] = g_loss
loss_dict["g_cooccur"] = g_cooccur_loss
loss_dict["rec_F1_H"] = recon_freq_loss_img1_high
loss_dict["rec_F1_L"] = recon_freq_loss_img1_low
loss_dict["rec_F2_H"] = recon_freq_loss_img2_structure

g_optim.zero_grad()
(recon_loss + g_loss + g_cooccur_loss + 
 recon_freq_loss_img1 + 
 args.w_high_recon * recon_freq_loss_img2_structure + 
 args.w_recon_fft * recon_fft + 
 args. w_fft_swap_H * fft_swap_H ).backward()
g_optim.step()

accumulate(e_ema, e_module, accum)
accumulate(g_ema, g_module, accum)

loss_reduced = reduce_loss_dict(loss_dict)

# d_loss_val = loss_reduced["d"].mean().item()
cooccur_val = loss_reduced["cooccur"].mean().item()
# recon_val = loss_reduced["recon"].mean().item()
# g_loss_val = loss_reduced["g"].mean().item()
g_cooccur_val = loss_reduced["g_cooccur"].mean().item()
r1_val = loss_reduced["r1"].mean().item()
cooccur_r1_val = loss_reduced["cooccur_r1"].mean().item()
rec_F1_H = loss_reduced["rec_F1_H"].mean().item()
rec_F1_L = loss_reduced["rec_F1_L"].mean().item()
rec_F2_H = loss_reduced["rec_F2_H"].mean().item()











