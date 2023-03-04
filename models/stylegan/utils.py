

import torch
import torch.nn as nn
import numpy as np

from .stylegan import Generator


class StyleGAN(nn.Module):
    def __init__(self, G=None, noises=None, size=1024, noise_mode='const', force_fp32=True, 
                 **load_generator_kwargs):
        nn.Module.__init__(self)


        self.size = size
        self.G = G or load_generator(**load_generator_kwargs)
        self.n_latents = int((np.log2(size)-1) * 2)
        
        if noises is not None:
            load_noises(self.G, noises)

        self.noise_mode = noise_mode

        self.forward_kwargs = {'noise_mode': self.noise_mode,
                               'force_fp32': force_fp32}

    def forward(self, *args, **kwargs):
        return self.gen_w(*args, **kwargs)

    def gen_z(self, z):
        return self.G(z, **self.forward_kwargs)

    def gen_w(self, w, **kwargs):
        w = w.to(self.device)

        if w.ndim == 2:
            w = w.unsqueeze(1)
        if w.size(1) == 1:
            w = w.repeat(1, self.n_latents, 1)

        self.forward_kwargs.update(**kwargs)
        return self.G.synthesis(w, input_type='w', **self.forward_kwargs)
    
    def generate_random_w(self, count: int=1, mode='w'):
        z = torch.randn(count, 512, device=self.device)
        w = self.G.mapping(z)[:, 0]
        if mode == 'w+':
            return w.unsqueeze(1).repeat(1, self.n_latents, 1)
        return w

    @torch.no_grad()
    def get_latent_stats(self, n_sample=100000):
        mean, std = self.sample_w(n_sample=n_sample)
        return mean, std
        
    def sample_w(self, n_sample=100000):
        z = torch.randn(n_sample, 512, device=self.device)
        w = self.G.mapping(z)[:,0]  # reduce broadcast dimension to a sinle one

        mean = w.mean(0, keepdim=True)
        std = w.std(0, keepdim=True)
        return mean, std

    @property
    def device(self):
        return next(self.parameters()).device


def configure_gan_noise(G, zero_out_noise=True, noises=None):
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    if zero_out_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.zeros_like(buf)
    elif not zero_out_noise and noises is None:
        for key in noise_bufs:
            noise_bufs[key].copy_(torch.randn_like(noise_bufs[key]))
    elif not zero_out_noise and noises is not None:
        load_noises(G, noises)
    else:
        raise ValueError('When passing a value to noises, the "zero_out_noise" flag must be set to False!')
    

@torch.no_grad()
def load_generator(to_01=True, zero_out_noise=False, noises=None, G_kwargs=None):
        
    G_kwargs = torch.load('checkpoints/G_kwargs.pt')
    G = Generator(to_01=to_01, **G_kwargs)
    
    G = G.eval()
    G.requires_grad_(False)
    configure_gan_noise(G, zero_out_noise=zero_out_noise, noises=noises)
    return G

def load_noises(G, noises):
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    for key in noise_bufs:
        noise_bufs[key].copy_(noises[key])

if __name__ == '__main__':
    gan = load_generator(to_01=False, size=256)
