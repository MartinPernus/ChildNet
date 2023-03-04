
import torch
import torch.nn as nn
from models.stylegan import StyleGAN
from models.e4e import E4e

class KinshipModel(nn.Module):
    def __init__(self, model_weights=None, resolution=1024, sample=False):
        super().__init__()
        assert model_weights in (None, 'fiw', 'nokdb')
        if model_weights is None:
            print('warning, model is not pretrained!')

        self.sample = sample
        self.decoder = StyleGAN(size=resolution, to_01=True)
        self.e4e = E4e(w_mode='w+', gan_size=resolution)

        self.decoder.eval().requires_grad_(False)
        self.e4e.eval().requires_grad_(False)

        _, std = self.decoder.get_latent_stats()
        self.gene_model = GeneModel(std)

        if model_weights is not None:
            checkpoint_file = f'checkpoints/childnet_{model_weights}.pt'
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            self.load_state_dict(checkpoint)

        if self.sample:
            self.eval_mode_with_sampling()
        if not self.sample:
            self.eval()

    def eval_mode_with_sampling(self):
        self.eval()
        for m in self.modules():
            if type(m) is nn.Dropout:
                m.train()
        return self

    def encoder_forward(self, img1, img2, move2parent=None):
        x1 = self.e4e(img1)
        x2 = self.e4e(img2)
        x, norm = self.gene_model(x1, x2, sample=self.sample, move2parent=move2parent)
        return x, norm

    def forward(self, father, mother, move2parent=None):
        child, _ = self.encoder_forward(father, mother, move2parent=move2parent)
        child = self.decoder(child)
        return child


class GeneModel(nn.Module):
    def __init__(self, w_std, merge_kwargs={}):
        super().__init__()

        self.merge_mu_att = StructuralMerge(merge_kwargs)
        self.merge_mu_res = StructuralMerge(merge_kwargs)
        
        self.register_buffer('w_std', w_std.view(1, 1, 512))  

    def forward(self, x1, x2, move2parent: float=None, sample=True):
        alpha = torch.sigmoid(self.merge_mu_att(x1, x2))
        if move2parent is not None:
            assert -1 <= move2parent <= 1
            if move2parent > 0:
                alpha = alpha + move2parent * (1-alpha)
            else:
                alpha = alpha + move2parent * alpha

        mu = alpha * x1 + (1-alpha) * x2

        mu_residual = self.merge_mu_res(x1, x2)
        mu_residual = mu_residual * self.w_std
        mu = mu + mu_residual

        mu_norm = torch.mean(mu_residual**2)
        x = mu
        return x, mu_norm


class StructuralMerge(nn.Module):
   def __init__(self, merge_kwargs={}):
       super().__init__()
       self.merge = {key: Merge(**merge_kwargs) for key in ('coarse', 'medium', 'fine')}
       self.merge = nn.ModuleDict(self.merge)

   def forward(self, x1, x2):
       x1_coarse = x1[:, :4, :]
       x2_coarse = x2[:, :4, :]

       x1_medium = x1[:, 4:8, :]
       x2_medium = x2[:, 4:8, :]

       x1_fine = x1[:, 8:, :]
       x2_fine = x2[:, 8:, :]

       x_coarse = self.merge['coarse'](x1_coarse, x2_coarse)
       x_medium = self.merge['medium'](x1_medium, x2_medium)
       x_fine = self.merge['fine'](x1_fine, x2_fine)

       x = torch.cat((x_coarse, x_medium, x_fine), dim=1)
       return x

class Merge(nn.Module):
    def __init__(self, dim=256, n_layers_first=5, n_layers_second=5, out_dim=512, dropout_p=0.5, multiple_alphas=True):
        super().__init__()

        if not multiple_alphas:
            out_dim = 1

        self.fc1 = SingleFC(dim, dropout_p=dropout_p, n_layers=n_layers_first)
        self.fc2 = SingleFC(dim, dropout_p=dropout_p, n_layers=n_layers_first)
        self.fc_final = FinalFC(2*dim, out_dim, dropout_p=dropout_p, n_layers=n_layers_second)

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc_final(x)
        return x


class SingleFC(nn.Module):
    def __init__(self, dim=256, n_layers=1, dropout_p=0.5):
        super().__init__()
        in_dim = 512
        layers = [LatentBatchNorm(in_dim),
                  nn.Linear(in_dim, dim),
                  nn.LeakyReLU()]

        for _ in range(n_layers-1):
            layers.extend([nn.Dropout(dropout_p),
                           LatentBatchNorm(dim),
                           nn.Linear(dim, dim),
                           nn.LeakyReLU()])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x

class FinalFC(nn.Module):
    def __init__(self, dim, out_dim, dropout_p=0.5, n_layers=1):
        super().__init__()
        self.dim = dim
        layers = [nn.Dropout(dropout_p)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(dim, dim),
                           nn.LeakyReLU(),
                           nn.Dropout(dropout_p),
                           LatentBatchNorm(dim),
                           ])
        layers.extend([nn.Linear(dim, out_dim)])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x

class LatentBatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.batch_norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.dim)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        return x