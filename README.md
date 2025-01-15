# ZS-IQA

## Foundation Models Boost Low-Level Perceptual Similarity Metrics

[Abhijay Ghildyal](https://abhijay9.github.io/), [Nabajeet Barman](https://www.linkedin.com/in/nabajeetbarman/), and [Saman Zadtootaghaj](https://www.linkedin.com/in/saman-zadtootaghaj-76947568/). In ICASSP, 2025. [[Arxiv]](https://arxiv.org/abs/2409.07650)

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