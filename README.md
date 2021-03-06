# ProbGAN: Towards Probabilistic GAN with Theoretical Guarantees
This repository contains the PyTorch implementation of the ProbGAN. 
This paper appears at ICLR 2019.
If you find this repo useful for your research, please consider citing our [[paper]](https://openreview.net/forum?id=H1l7bnR5Ym).

```
@article{he2018probgan,
  title={ProbGAN: Towards Probabilistic GAN with Theoretical Guarantees},
  author={He, Hao and Wang, Hao and Lee, Guang-He and Tian, Yonglong},
  year={2018}
}
```

## Results

![Result Image](./figures/results_image.png)


## Install
This codebase is tested with Ubuntu 16.04 LTS, Python 3.6.8, PyTorch 1.0.0, and CUDA 9.0.

## Usage

To train ProbGAN on different dataset with different GAN objectives.

    GAN objectives
    NS: original GAN (Non-saturating version)
    MM: original GAN (Min-max version)
    W: Wasserstein GAN
    LS: Least-Square GAN

```bash
python train.py --dataset [cifar10 | stl10] --gan_obj [NS | MM | W | LS]
```
## Acknowledgement
We inspired by the code of [Bayesian GAN](https://github.com/andrewgordonwilson/bayesgan) to train probabilistic GAN with SGHMC. 