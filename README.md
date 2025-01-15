# ZS-IQA
This is the official repository of the paper tiled
## Foundation Models Boost Low-Level Perceptual Similarity Metrics

[Abhijay Ghildyal](https://abhijay9.github.io/), [Nabajeet Barman](https://www.linkedin.com/in/nabajeetbarman/), and [Saman Zadtootaghaj](https://www.linkedin.com/in/saman-zadtootaghaj-76947568/). 

In ICASSP, 2025. Please checkout the paper on [[Arxiv]](https://arxiv.org/abs/2409.07650)

## Abstract

For full-reference image quality assessment (FR-IQA) using deep-learning approaches, the perceptual similarity score between a distorted image and a reference image is typically computed as a distance measure between features extracted from a pretrained CNN or more recently, a Transformer network. Often, these intermediate features require further fine-tuning or processing with additional neural network layers to align the final similarity scores with human judgments. So far, most IQA models based on foundation models have primarily relied on the final layer or the embedding for the quality score estimation. In contrast, this work explores the potential of utilizing the intermediate features of these foundation models, which have largely been unexplored so far in the design of low-level perceptual similarity metrics. We demonstrate that the intermediate features are comparatively more effective. Moreover, without requiring any training, these metrics can outperform both traditional and state-of-the-art learned metrics by utilizing distance measures between the features.

## Using the ZS-IQA Model

### Install

```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install git+https://github.com/openai/CLIP.git
pip install requirements.txt
```

### Run evaluation

```
python run.py -dat <dataset(pipal)> -m <method_name(clip_vitb32,dinov1,embed_clip_vitb32,embed_dinov1)> -d <distance(l2,cos,wsd,jsd,skld)> -s <saveas>
```

for robustness tests:

```
python run.py -dat <dataset(pipal)> -m <method_name(clip_vitb32,dinov1,embed_clip_vitb32,embed_dinov1)> -d <distance(l2,cos,wsd,jsd,skld)> -s <saveas> -rob <tra,sca,rot> -pct <pctPixels>
```

Feel free to experiment with different backbones <b>clip_vitb32, dinov1, embed_clip_vitb32, and embed_dinov1</b>. For a complete list of backbones and methods, please refer to the code available in models.py 

Feel free to experiment with different distances, such as L2 (<b>l2</b>), cosine (<b>cos</b>), Wasserstein distance, Jensen-shannon distance (<b>jsd</b>), and Symmetric KL-divergence (<b>skld</b>).

### Run single image evaluation

```
python run_single_img.py -m dinov1 -d l2 -ref <path_to_pipal_dataset>/ref/A0001.bmp -dis <path_to_pipal_dataset>/dis/A0001_00_00.bmp
```

## Citation

If you find this repository useful for your research, please use the following.

```
@inproceedings{ghildyal2025zsiqa,
  title={Foundation Models Boost Low-Level Perceptual Similarity Metrics},
  author={Abhijay Ghildyal and Nabajeet Barman and Saman Zadtootaghaj},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  year={2025}
}
```

## Acknowledgements
This repository borrows from [Deep-network-based-distribution-measures-for-full-reference-image-quality-assessment
](https://github.com/Buka-Xing/Deep-network-based-distribution-measures-for-full-reference-image-quality-assessment/tree/main). We thank the authors of these repositories for their incredible work and inspiration.
