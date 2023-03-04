# ChildNet: Structural Kin-based Facial Synthesis Model with Appearance Control Mechanisms
## More Information Coming Soon!

## Requirements
This repository depends on [E4e](https://github.com/omertov/encoder4editing) and [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) repositories. Setting up these repositories requires C++/CUDA compilation, which can sometimes cause issues. 

If experiencing problems, try using [this dockerfile](https://github.com/rosinality/alias-free-gan-pytorch/blob/main/Dockerfile).


## Download Models
`./download.sh`

## Example Usage (Inference)
`python main.py --father imgs/father.jpg --mother imgs/mother.jpg`

The image result (synthesised child) is saved under `imgs/result.jpg`.

Refer to `main.py` to find out about further argument options. Further documentation will be provided soon.


## Code Acknowledgements
[Encoder for Editing](https://github.com/omertov/encoder4editing) 

[StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) 

