import torch
import torch.nn as nn


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


class SingleFC(nn.Module):
    def __init__(self, in_dim=512, hid_dim=256, out_dim=256, n_layers=1, dropout_p=0.0, batch_norm=True):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_dim),
                  nn.LeakyReLU()]

        for i in range(n_layers-1):
            if i != (n_layers - 2):  # last index
                i_out_dim = hid_dim
            else:
                i_out_dim = out_dim

            layers.extend([nn.Dropout(dropout_p),
                            LatentBatchNorm(hid_dim) if batch_norm else nn.Identity(),
                            nn.Linear(hid_dim, i_out_dim),
                            nn.LeakyReLU()])  # nonlinear can be last since we feed this to finalfc anyway
                
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x

class FinalFC(nn.Module):
    def __init__(self, in_dim=256, hid_dim=256, dropout_p=0.0, n_layers=1, batch_norm=True):
        super().__init__()
        if n_layers == 1:
            layers = [nn.Dropout(dropout_p), LatentBatchNorm(), nn.Linear(in_dim, 512)]
        else:
            layers = self.construct_layers(in_dim, hid_dim, dropout_p, n_layers, batch_norm=batch_norm)
        self.main = nn.ModuleList(layers)

    def construct_layers(self, in_dim, hid_dim, dropout_p, n_layers, batch_norm):
        layers = [nn.Dropout(dropout_p), 
                  LatentBatchNorm(in_dim) if batch_norm else nn.Identity(), 
                  nn.Linear(in_dim, hid_dim), 
                  nn.LeakyReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Dropout(dropout_p),
                           LatentBatchNorm(hid_dim) if batch_norm else nn.Identity(),
                           nn.Linear(hid_dim, hid_dim),
                           nn.LeakyReLU(),
                           ])
        layers.extend([nn.Dropout(dropout_p), 
                       LatentBatchNorm(hid_dim) if batch_norm else nn.Identity(),
                       nn.Linear(hid_dim, 512)])
        return layers

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Merge(nn.Module):
    def __init__(self, in_dim1=512, in_dim2=64, hid_dim=128, n_layers_first=3, n_layers_second=3, dropout_p=0.5,
                 batch_norm=True):
        super().__init__()
        self.fc1 = SingleFC(in_dim=in_dim1, hid_dim=hid_dim, out_dim=hid_dim, dropout_p=dropout_p, 
                            n_layers=n_layers_first, batch_norm=batch_norm)
        self.fc2 = SingleFC(in_dim=in_dim2, hid_dim=hid_dim, out_dim=hid_dim, dropout_p=dropout_p, 
                            n_layers=n_layers_first, batch_norm=batch_norm)
        self.fc_final = FinalFC(in_dim=2*hid_dim, hid_dim=hid_dim, dropout_p=dropout_p, n_layers=n_layers_second,
                                batch_norm=batch_norm)

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc_final(x)
        return x 


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
        
class AgeGenderDisentanglement(nn.Module):   
    def __init__(self, mean=None, std=None, emb_dim=128, hid_dim=256, batch_norm=False, n_age=10):
        super().__init__()
        merge_kwargs = dict(in_dim1=512, in_dim2=2*emb_dim, hid_dim=hid_dim, 
                            n_layers_first=3, n_layers_second=3, 
                            dropout_p=0, batch_norm=batch_norm)

        self.mapping = StructuralMerge(merge_kwargs)
        if mean is None:
            mean = std = torch.randn(1, 1, 512)

        self.register_buffer('latent_mean', mean.view(1,1,512))
        self.register_buffer('latent_std', std.view(1,1,512))

        self.embedding_age = nn.Embedding(num_embeddings=n_age, embedding_dim=emb_dim)
        self.embedding_gender = nn.Embedding(num_embeddings=2, embedding_dim=emb_dim)

        ckpt = torch.load('checkpoints/disentanglement.pt', map_location='cpu')
        self.load_state_dict(ckpt)

    def latent_forward(self, w, embedding):
        embedding = embedding.unsqueeze(1).repeat(1, w.size(1), 1)
        w_in = (w - self.latent_mean) / self.latent_std
        w_delta = self.mapping(w_in, embedding)
        w_delta = w_delta * self.latent_std  
        w = w + w_delta
        return w, w_delta

    def forward(self, w, age, gender):  
        embedding = self.embed_age_and_gender(age, gender)
        w, w_delta = self.latent_forward(w, embedding)
        return w, w_delta

    def embed_age_and_gender(self, age, gender):
        embedding_age = self.embedding_age(age.long())
        embedding_gender = self.embedding_gender(gender.long())
        embedding = torch.cat((embedding_age, embedding_gender), dim=1)
        return embedding