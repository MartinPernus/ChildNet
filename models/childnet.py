import torch.nn as nn
from models.disentanglement import AgeGenderDisentanglement
from models.kinship_model import KinshipModel

class ChildNet(nn.Module):
    def __init__(self, model_weights, sample=False):
        super().__init__()
        assert model_weights in ('nokdb', 'fiw')
        self.kinship_model = KinshipModel(model_weights=model_weights, sample=sample).eval()
        self.disentangle = AgeGenderDisentanglement(batch_norm=False).eval()

    def forward(self, img_father, img_mother, age=None, gender=None, move2parent=None):
        w, _ = self.kinship_model.encoder_forward(img_father, img_mother, move2parent=move2parent)
        if age is not None and gender is not None:
            w, _ = self.disentangle(w, age, gender)
        img = self.kinship_model.decoder(w)
        return img