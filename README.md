# ChildNet: Structural Kin-based Facial Synthesis Model with Appearance Control Mechanisms


## Installation
`cat requirements.txt | xargs -n 1 -L 1 pip install`

## Download Models
`./download.sh`

## Example Usage (Inference)
`python main.py --input_dir imgs/input --description "long sleeve silk crepe de chine shirt featuring graphic pattern printed in tones of blue"`

The `--input_dir` argument specifies directory of images (256x256 resolution) to be edited.
 Nevertheless, finding a good dataset for such segmentation training could be a problem!


## Code Acknowledgements
[Encoder for Editing](https://github.com/omertov/encoder4editing) 

[StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) 


## Sponsor Acknowledgements
Supported in parts by the Slovenian Research Agency ARRS through the Research Programme P2-0250(B) Metrology and Biometric System, the ARRS Project J2-2501(A) DeepBeauty and the ARRS junior researcher program.

<img src=imgs/ARRSLogo.png width="400">

