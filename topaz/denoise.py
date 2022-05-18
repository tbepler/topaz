from __future__ import absolute_import, print_function, division

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


from topaz.utils.data.loader import load_image
from topaz.filters import AffineFilter, AffineDenoise, GaussianDenoise, gaussian_filter, inverse_filter



def denoise(model, x, patch_size=-1, padding=128):

    # check the patch plus padding size
    use_patch = False
    if patch_size > 0:
        s = patch_size + padding
        use_patch = (s < x.size(0)) or (s < x.size(1))

    if use_patch:
        return denoise_patches(model, x, patch_size, padding=padding)

    with torch.no_grad():
        x = x.unsqueeze(0).unsqueeze(0)
        y = model(x).squeeze()

    return y



def denoise_image(mic, models, lowpass=1, cutoff=0, gaus=None, inv_gaus=None, deconvolve=False
                 , deconv_patch=1, patch_size=-1, padding=0, normalize=False
                 , use_cuda=False):
    if lowpass > 1:
        mic = dn.lowpass(mic, lowpass)

    mic = torch.from_numpy(mic)
    if use_cuda:
        mic = mic.cuda()

    # normalize and remove outliers
    mu = mic.mean()
    std = mic.std()
    x = (mic - mu)/std
    if cutoff > 0:
        x[(x < -cutoff) | (x > cutoff)] = 0

    # apply guassian/inverse gaussian filter
    if gaus is not None:
        x = dn.denoise(gaus, x)
    elif inv_gaus is not None:
        x = dn.denoise(inv_gaus, x)
    elif deconvolve:
        # estimate optimal filter and correct spatial correlation
        x = dn.correct_spatial_covariance(x, patch=deconv_patch)

    # denoise
    mic = 0
    for model in models:
        mic += dn.denoise(model, x, patch_size=patch_size, padding=padding)
    mic /= len(models)

    # restore pixel scaling
    if normalize:
        mic = (mic - mic.mean())/mic.std()
    else:
        # add back std. dev. and mean
        mic = std*mic + mu

    # back to numpy/cpu
    mic = mic.cpu().numpy()

    return mic


def denoise_patches(model, x, patch_size, padding=128):
    y = torch.zeros_like(x)
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for i in range(0, x.size(2), patch_size):
            for j in range(0, x.size(3), patch_size):
                # include padding extra pixels on either side
                si = max(0, i - padding)
                ei = min(x.size(2), i + patch_size + padding)

                sj = max(0, j - padding)
                ej = min(x.size(3), j + patch_size + padding)

                xij = x[:,:,si:ei,sj:ej]
                yij = model(xij).squeeze() # denoise the patch

                # match back without the padding
                si = i - si
                sj = j - sj

                y[i:i+patch_size,j:j+patch_size] = yij[si:si+patch_size,sj:sj+patch_size]

    return y

def denoise_stack(model, stack, batch_size=20, use_cuda=False):
    denoised = np.zeros_like(stack)
    with torch.no_grad():
        stack = torch.from_numpy(stack).float()

        for i in range(0, len(stack), batch_size):
            x = stack[i:i+batch_size]
            if use_cuda:
                x = x.cuda()
            mu = x.view(x.size(0), -1).mean(1)
            std = x.view(x.size(0), -1).std(1)
            x = (x - mu.unsqueeze(1).unsqueeze(2))/std.unsqueeze(1).unsqueeze(2)

            y = model(x.unsqueeze(1)).squeeze(1)
            y = std.unsqueeze(1).unsqueeze(2)*y + mu.unsqueeze(1).unsqueeze(2)

            y = y.cpu().numpy()
            denoised[i:i+batch_size] = y

    return denoised

def spatial_covariance(x, n=11, s=11):

    # first, calculate the mean of each region
    #f = torch.ones(n, n)/(n*n)
    #f = f.to(x.device)
    #mu = F.conv2d(x.unsqueeze(0).unsqueeze(1), f.unsqueeze(0).unsqueeze(1)).squeeze()
    
    # now, to calculate the spatial covariance, we calculate
    # the expectation of (x_ii-mu)*(x_jk-mu)

    p = n//2

    """
    x_c = x[p:-p,p:-p] - mu
    cov = torch.zeros(n, n).to(x.device)
    N = x.size(0) - n + 1
    M = x.size(1) - n + 1
    for i in range(n):
        for j in range(n):
            x_ij = x[i:i+N,j:j+M] - mu[i,j]
            cov[i,j] = torch.mean(x_c*x_ij)
    """

    x_c = x[p:-p,p:-p]
    cov = F.conv2d(x.unsqueeze(0).unsqueeze(1), x_c.unsqueeze(0).unsqueeze(1)).squeeze()
    cov /= x_c.size(0)*x_c.size(1)

    return cov

def spatial_covariance_old(x, n=11, s=11):
    tiles = []
    for i in range(0, x.shape[0], s):
        for j in range(0, x.shape[1], s):
            if i+n <= x.shape[0] and j+n <= x.shape[1]:
                t = x[i:i+n,j:j+n]
                tiles.append(t)
    tiles = torch.stack(tiles, 0)
    tiles = tiles.view(len(tiles), -1)
    m = tiles.mean(1, keepdim=True)
    tiles = tiles - m

    m = tiles.mean(0)
    x = (tiles - m)
    cov = torch.mm(x.t(), x)/x.size(0)
    cov = cov.view(n, n, n, n)

    # we are only interested in the central pixel
    i = n//2
    return cov[i,i]

def estimate_unblur_filter(x, width=11, s=11):
    """
    Estimate parameters of the affine filter that would give
    zero autocovariance to the image.
    """

    cov = spatial_covariance(x, n=width, s=s)

    # calculate the power spectrum, this is the fft of the autocovariance
    cov = cov.cpu().numpy()
    ps = np.fft.ifftshift(cov)
    ps = np.fft.fft2(ps)
    # clip ps <= 0 to 1...
    ps.real[ps.real <= 0] = 1
    # also set the zero frequency term to 1
    ps[0,0] = 1

    # calculate the filter that flattens the power spectrum
    # this corrects the autocorrelation

    F = 1/np.sqrt(ps.real)
    w_inv = np.fft.fftshift(np.fft.ifft2(F)).real

    return AffineFilter(w_inv), cov

def estimate_unblur_filter_gaussian(x, width=11, s=11):
    """
    Estimate parameters of the Gaussian filter that would give
    the closest spatial covariance structure to that observed
    in image x if the unfiltered image had zero spatial covariance.
    Then, return inverse of that filter.
    """

    from scipy.signal import convolve2d
    from scipy.optimize import minimize

    cov = spatial_covariance(x, n=width, s=s)

    dim = s//2
    xx,yy = np.meshgrid(np.arange(-dim, dim+1), np.arange(-dim, dim+1))
    d = xx**2 + yy**2
    d = torch.from_numpy(d).float().to(cov.device)

    # solve for Gaussian kernel parameters that give closest covariance structure
    def loss(params):
        params = torch.from_numpy(params).float().to(cov.device)
        params.requires_grad = True
        params_exp = torch.exp(params)
        sigma = params_exp[0]
        alpha = params_exp[1]

        w = alpha*torch.exp(-0.5*d/sigma**2)
        w = w.unsqueeze(0).unsqueeze(1)

        c_w = F.conv2d(w, w, padding=width//2).squeeze() # covaraince structure given by application of w

        err = torch.sum((c_w - cov)**2)

        err.backward()
        grad = params.grad.data.cpu().numpy()
        err = err.item()

        return err, grad

    w0 = np.array([0, 0])
    result = minimize(loss, w0, jac=True)

    #return result, cov

    sigma = np.exp(result.x[0])
    alpha = np.exp(result.x[1])

    w = gaussian_filter(sigma)*alpha
    w_inv = inverse_filter(w)

    return AffineFilter(w_inv), sigma, alpha, cov

def correct_spatial_covariance(x, width=11, s=11, patch=1):
    """
    Estimates the spatial covariance in the micrograph, finds closest Guassian kernel
    parameters that would give this covariance structure, and applies inverse of that
    filter to correct the spatial covariance.
    """
    if patch > 1:

        N = [x.size(0)//patch]*patch
        for i in range(x.size(0)%patch):
            N[i] += 1
        M = [x.size(1)//patch]*patch
        for i in range(x.size(1)%patch):
            M[i] += 1

        y = torch.zeros_like(x)
        
        i = 0
        for n in N:
            j = 0
            for m in M:
                pad_ii = max(0, i-width//2)
                pad_ij = min(x.size(0), i+n+width//2)

                pad_ji = max(0, j-width//2)
                pad_jj = min(x.size(1), j+m+width//2)

                x_ij = x[pad_ii:pad_ij,pad_ji:pad_jj]
                y_ij = correct_spatial_covariance(x_ij, width=width, s=s)
                y[i:i+n,j:j+m] = y_ij[i-pad_ii:i-pad_ii+n,j-pad_ji:j-pad_ji+m]

                j += m
            i += n

    else:
        # estimate the parameters
        #f,sigma,alpha,cov = estimate_unblur_filter(x, width=width, s=s)
        f,cov = estimate_unblur_filter(x, width=width, s=s)
        f = f.to(x.device)

        x = x.unsqueeze(0).unsqueeze(1)
        y = f(x).squeeze()

    return y



class GaussianNoise:
    def __init__(self, x, sigma=1.0, crop=500, xform=True):
        self.x = x
        self.sigma = sigma
        self.crop = crop
        self.xform = xform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]

        # randomly crop
        if self.crop is not None:
            size = self.crop

            n,m = x.shape
            i = np.random.randint(n-size+1)
            j = np.random.randint(m-size+1)

            x = x[i:i+size, j:j+size]

        # randomly flip
        if self.xform:
            if np.random.rand() > 0.5:
                x = np.flip(x, 0)
            if np.random.rand() > 0.5:
                x = np.flip(x, 1)

            k = np.random.randint(4)
            x = np.rot90(x, k=k)

        # generate random noise
        std = x.std()*self.sigma
        n,m = x.shape
        r1 = np.random.randn(n, m).astype(np.float32)*std
        r2 = np.random.randn(n, m).astype(np.float32)*std

        return x+r1, x+r2
    

class L0Loss:
    def __init__(self, eps=1e-8, gamma=2):
        self.eps = eps
        self.gamma = gamma

    def __call__(self, x, y):
        return torch.mean((torch.abs(x - y) + self.eps)**self.gamma)


def eval_noise2noise(model, dataset, criteria, batch_size=10
                    , use_cuda=False, num_workers=0):
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size
                                               , num_workers=num_workers)

    n = 0
    loss = 0

    model.eval()
        
    with torch.no_grad():
        for x1,x2 in data_iterator:
            if use_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()

            x1 = x1.unsqueeze(1)
            y = model(x1).squeeze(1)

            loss_ = criteria(y, x2).item()

            b = x1.size(0)
            n += b
            delta = b*(loss_ - loss)
            loss += delta/n

    return loss


def train_noise2noise(model, dataset, lr=0.001, optim='adagrad', batch_size=10, num_epochs=100
                     , criteria=nn.MSELoss(), dataset_val=None
                     , use_cuda=False, num_workers=0, shuffle=True):

    gamma = None
    if criteria == 'L0':
        gamma = 2
        eps = 1e-8
        criteria = L0Loss(eps=eps, gamma=gamma)
    elif criteria == 'L1':
        criteria = nn.L1Loss()
    elif criteria == 'L2':
        criteria = nn.MSELoss()
    
    if optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle
                                               , num_workers=num_workers)

    total = len(dataset)

    for epoch in range(1, num_epochs+1):
        model.train()
        
        n = 0
        loss_accum = 0

        if gamma is not None:
            # anneal gamma to 0
            criteria.gamma = 2 - (epoch-1)*2/num_epochs

        for x1,x2 in data_iterator:
            if use_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()

            x1 = x1.unsqueeze(1)
            y = model(x1).squeeze(1)

            loss = criteria(y, x2)
            
            loss.backward()
            optim.step()
            optim.zero_grad()

            loss = loss.item()
            b = x1.size(0)

            n += b
            delta = b*(loss - loss_accum)
            loss_accum += delta/n

            print('# [{}/{}] {:.2%} loss={:.5f}'.format(epoch, num_epochs, n/total, loss_accum)
                 , file=sys.stderr, end='\r')
        print(' '*80, file=sys.stderr, end='\r')

        if dataset_val is not None:
            loss_val = eval_noise2noise(model, dataset_val, criteria
                                       , batch_size=batch_size
                                       , num_workers=num_workers
                                       , use_cuda=use_cuda
                                       )
            yield epoch, loss_accum, loss_val
        else:
            yield epoch, loss_accum


def eval_mask_denoise(model, dataset, criteria, p=0.01 # masking rate
                     , batch_size=10, use_cuda=False, num_workers=0):
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size
                                               , num_workers=num_workers)

    n = 0
    loss = 0

    model.eval()
        
    with torch.no_grad():
        for x in data_iterator:
            # sample the mask
            mask = (torch.rand(x.size()) < p)
            r = torch.randn(x.size())

            if use_cuda:
                x = x.cuda()
                mask = mask.cuda()
                r = r.cuda()

            # mask out x by replacing from N(0,1)
            x_ = mask.float()*r + (1-mask.float())*x

            # denoise the image
            y = model(x_.unsqueeze(1)).squeeze(1)

            # calculate the loss for the masked entries
            x = x[mask]
            y = y[mask]

            loss_ = criteria(y, x).item()

            b = x.size(0)
            n += b
            delta = b*(loss_ - loss)
            loss += delta/n

    return loss


def train_mask_denoise(model, dataset, p=0.01, lr=0.001, optim='adagrad', batch_size=10, num_epochs=100
                      , criteria=nn.MSELoss(), dataset_val=None
                      , use_cuda=False, num_workers=0, shuffle=True):

    gamma = None
    if criteria == 'L0':
        gamma = 2
        eps = 1e-8
        criteria = L0Loss(eps=eps, gamma=gamma)
    elif criteria == 'L1':
        criteria = nn.L1Loss()
    elif criteria == 'L2':
        criteria = nn.MSELoss()
    
    if optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    data_iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle
                                               , num_workers=num_workers)

    total = len(dataset)

    for epoch in range(1, num_epochs+1):
        model.train()
        
        n = 0
        loss_accum = 0

        if gamma is not None:
            # anneal gamma to 0
            criteria.gamma = 2 - (epoch-1)*2/num_epochs

        for x in data_iterator:
            b = x.size(0)

            # sample the mask
            mask = (torch.rand(x.size()) < p)
            r = torch.randn(x.size())

            if use_cuda:
                x = x.cuda()
                mask = mask.cuda()
                r = r.cuda()

            # mask out x by replacing from N(0,1)
            x_ = mask.float()*r + (1-mask.float())*x

            # denoise the image
            y = model(x_.unsqueeze(1)).squeeze(1)

            # calculate the loss for the masked entries
            x = x[mask]
            y = y[mask]

            loss = criteria(y, x)
            
            loss.backward()
            optim.step()
            optim.zero_grad()

            loss = loss.item()
            n += b
            delta = b*(loss - loss_accum)
            loss_accum += delta/n

            print('# [{}/{}] {:.2%} loss={:.5f}'.format(epoch, num_epochs, n/total, loss_accum)
                 , file=sys.stderr, end='\r')
        print(' '*80, file=sys.stderr, end='\r')

        if dataset_val is not None:
            loss_val = eval_mask_denoise(model, dataset_val, criteria, p=p
                                        , batch_size=batch_size
                                        , num_workers=num_workers
                                        , use_cuda=use_cuda
                                        )
            yield epoch, loss_accum, loss_val
        else:
            yield epoch, loss_accum


def lowpass(x, factor=1, dims=2):
    """ low pass filter with FFT """

    if dims == 2:
        freq0 = np.fft.fftfreq(x.shape[-2])
        freq1 = np.fft.rfftfreq(x.shape[-1])
    elif dims == 3:
        freq0 = np.fft.fftfreq(x.shape[-3])
        freq1 = np.fft.fftfreq(x.shape[-2])
        freq2 = np.fft.rfftfreq(x.shape[-1])

    freq = np.meshgrid(freq0, freq1, indexing='ij') if dims ==2 else np.meshgrid(freq0, freq1, freq2, indexing='ij')
    freq = np.stack(freq, dims)

    r = np.abs(freq)
    mask = np.any((r > 0.5/factor), dims) 

    F = np.fft.rfftn(x)
    F[...,mask] = 0

    f = np.fft.irfftn(F, s=x.shape)
    f = f.astype(x.dtype)

    return f


def gaussian(x, sigma=1, scale=5, use_cuda=False, dims=2):
    """
    Apply Gaussian filter with sigma to image. Truncates the kernel at scale times sigma pixels
    """

    f = GaussianDenoise(sigma, scale=scale, dims=dims)
    if use_cuda:
        f.cuda()

    with torch.no_grad():
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        if use_cuda:
            x = x.cuda()
        y = f(x).squeeze().cpu().numpy()
    return y




###########################################################
# new stuff below
###########################################################

class Denoise():
    def __init__(self, model:torch.nn.Module):
         self.model = model
         
    def __repr__(self) -> str:
        pass
    
    def __call__(self, input):
        self.denoise(input)
    
    def train(self):
        pass
    
    def denoise(self):
        pass
    
  
#main classes  
class Denoise2D(Denoise):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        
    def train(self, data):
        pass
    
    def denoise(self, data):    
        pass
  
    
class Denoise3D(Denoise):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        
    def train(self, data):
        pass
    
    def denoise(self):
        pass
    
    

# 3D denoising  
def denoise(model, path, outdir, suffix, patch_size=128, padding=128, batch_size=1
           , volume_num=1, total_volumes=1):
    with open(path, 'rb') as f:
        content = f.read()
    tomo,header,extended_header = mrc.parse(content)
    tomo = tomo.astype(np.float32)
    name = os.path.basename(path)

    mu = tomo.mean()
    std = tomo.std()
    # denoise in patches
    d = next(iter(model.parameters())).device
    denoised = np.zeros_like(tomo)

    with torch.no_grad():
        if patch_size < 1:
            x = (tomo - mu)/std
            x = torch.from_numpy(x).to(d)
            x = model(x.unsqueeze(0).unsqueeze(0)).squeeze().cpu().numpy()
            x = std*x + mu
            denoised[:] = x
        else:
            patch_data = PatchDataset(tomo, patch_size, padding)
            total = len(patch_data)
            count = 0

            batch_iterator = torch.utils.data.DataLoader(patch_data, batch_size=batch_size)
            for index,x in batch_iterator:
                x = x.to(d)
                x = (x - mu)/std
                x = x.unsqueeze(1) # batch x channel

                # denoise
                x = model(x)
                x = x.squeeze(1).cpu().numpy()

                # restore original statistics
                x = std*x + mu

                # stitch into denoised volume
                for b in range(len(x)):
                    i,j,k = index[b]
                    xb = x[b]

                    patch = denoised[i:i+patch_size,j:j+patch_size,k:k+patch_size]
                    pz,py,px = patch.shape

                    xb = xb[padding:padding+pz,padding:padding+py,padding:padding+px]
                    denoised[i:i+patch_size,j:j+patch_size,k:k+patch_size] = xb

                    count += 1
                    print('# [{}/{}] {:.2%}'.format(volume_num, total_volumes, count/total), name, file=sys.stderr, end='\r')

            print(' '*100, file=sys.stderr, end='\r')


    ## save the denoised tomogram
    if outdir is None:
        # write denoised tomogram to same location as input, but add the suffix
        if suffix is None: # use default
            suffix = '.denoised'
        no_ext,ext = os.path.splitext(path)
        outpath = no_ext + suffix + ext
    else:
        if suffix is None:
            suffix = ''
        no_ext,ext = os.path.splitext(name)
        outpath = outdir + os.sep + no_ext + suffix + ext

    # use the read header except for a few fields
    header = header._replace(mode=2) # 32-bit real
    header = header._replace(amin=denoised.min())
    header = header._replace(amax=denoised.max())
    header = header._replace(amean=denoised.mean())

    with open(outpath, 'wb') as f:
        mrc.write(f, denoised, header=header, extended_header=extended_header)
