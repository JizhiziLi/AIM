<h1 align="center">Deep Automatic Natural Image Matting [IJCAI-21]</h1>

<p align="center">
<a href="https://arxiv.org/abs/2107.07235"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
<a href=""><img  src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/2107.07235">Deep Automatic Natural Image Matting</a>.</h4>

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#network">Network</a> |
  <a href="#aim-500">AIM-500</a> |
  <a href="#results">Results</a> |
  <a href="#statement">Statement</a>
</p>


***
><h3><strong><i>📆 News</i></strong></h3>
> The training code, inference code and the pretrained models will be released soon. 
> 
> [2021-07-16]: Publish the validation dataset [AIM-500](https://drive.google.com/drive/folders/1IyPiYJUp-KtOoa-Hsm922VU3aCcidjjz?usp=sharing). Please follow the `readme.txt` for details.

## Introduction

<p align="justify">Different from previous methods only focusing on images with <em><u>salient opaque</u></em> foregrounds such as humans and animals, in this paper, we investigate the
difficulties when extending the automatic matting methods to natural images with <em><u>salient
transparent/meticulous</u></em> foregrounds or <em><u>non-salient</u></em> foregrounds.</p>

<p align="justify">To address the problem, we propose a novel end-to-end matting network, which can predict a generalized trimap for any image of the above types as a unified semantic representation. Simultaneously, the learned semantic features guide the matting network to focus on the transition areas via an attention mechanism.</p>

<p align="justify">We also construct a test set <strong>AIM-500</strong> that contains 500 diverse natural images covering all types along with manually labeled alpha mattes, making it feasible to benchmark the generalization ability of AIM models. Results of the experiments demonstrate that our network trained on available composite matting datasets outperforms existing methods both objectively and subjectively.</p>

## Network

![](demo/network.png)

We propose the methods consist of:

- <strong>Improved Backbone for Matting</strong>: an advanced max-pooling version of ResNet-34, serves as the backbone for the matting network, pretrained on ImageNet;

- <strong>Unified Semantic Representation</strong>: a type-wise semantic representation to replace the traditional trimaps;

- <strong>Guided Matting Process</strong>: an attention based mechanism to guide the matting process by leveraging the learned
semantic features from the semantic decoder to focus on extracting details only within transition area.

The backbone pretrained on ImageNet and the model pretrained on synthetic matting dataset will be released soon.

| Pretrained-backbone | Pretrained-model |
| :----:| :----: | 
|coming soon|coming soon| 


## AIM-500
We propose <strong>AIM-500</strong> (Automatic Image Matting-500), the first natural image matting test set, which contains 500 high-resolution real-world natural images from all three types (SO, STM, NS), many categories, and the manually labeled alpha mattes. Some examples and the amount of each category are shown below. The <strong>AIM-500</strong> dataset is <strong>published</strong> now, can be downloaded directly from [<u>this link</u>](https://drive.google.com/drive/folders/1IyPiYJUp-KtOoa-Hsm922VU3aCcidjjz?usp=sharing). Please follow the `readme.txt` for more details.

| Portrait | Animal | Transparent | Plant | Furniture | Toy | Fruit |
| :----:| :----: |  :----: |  :----: |  :----: |  :----: |  :----: | 
| 100 | 200 | 34 | 75 | 45 | 36 | 10 | 

![](demo/aim500.jpg)

## Results

We test our network on different types of images in AIM-500 and compare with previous SOTA methods, the results are shown below.

![](demo/exp.jpg)

## Statement

If you are interested in our work, please consider citing the following:
```
@inproceedings{ijcai2021-danim,
  title     = {Deep Automatic Natural Image Matting},
  author    = {Li, Jizhizi and Zhang, Jing and Tao, Dacheng},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  year      = {2021},
}
```

This project is under the MIT license. For further questions, please contact [jili8515@uni.sydney.edu.au](mailto:jili8515@uni.sydney.edu.au).

## Relevant Projects

[End-to-end Animal Image Matting](https://github.com/JizhiziLi/animal-matting)
<br><em>Jizhizi Li, Jing Zhang, Stephen J. Maybank, Dacheng Tao</em>

