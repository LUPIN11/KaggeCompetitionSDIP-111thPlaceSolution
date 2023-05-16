# 111th Place Solution

## Overview

I'd like to attribute a great portion of my bronze medal to the use of ensembling, which gives me a significant rank lift. Therefore, I will mainly introduce the ensembling step and then key points of each model ensembled. Lastly, some issues I discovered during the competition, which are the greatest fulfillment for me rather than the rank, will be discussed.

## Ensembling

The ensembling approach is very simple: do a weighted sum on the outputs of  models. This is to adjust the directions of the predicted embeddings to match the true directions as well as possible, considering that the score is average cosine similarity.

There are some principles:

1. better solo model has a higher ratio and makes the ensembling result better
2. search the best ratios of 2 models first, and then increase the number of models 1 by 1. By this way, there is always 1 variable only to handle.
3. The function $score=f(ratio)$ initially increases and then decreases, so we can use binary search. 

| model                         | score   | ratio |
| :---------------------------- | ------- | ----- |
| openai/clip-vit-large-patch14 | 0.55603 | 1     |
| SDIP CLIP kNNRegression       | 0.55313 | 1     |
| vit_large_patch16_384         | 0.53796 | 0.4   |
| CLIP Interrogator             | 0.45836 | 0.2   |

## Models

Pre-trained models that have more parameters and more relevant pretraining tasks can achieve a higher score, eg. openai/clip-vit-large-patch14 and vit_large_patch16_384.

Roughly speaking, SDIP CLIP kNNRegression and CLIP Interrogator are based on retrieval in the encoding space of CLIP. Models like this are very ingenious in that you don't train them and could try to implement some insightful prior knowledge.

> SDIP CLIP kNNRegression: https://www.kaggle.com/code/motono0223/sdip-clip-knnregression-zeroshot-method
>
> CLIP Interrogator: https://www.kaggle.com/code/leonidkulyk/lb-0-45836-blip-clip-clip-interrogator

As for training(fine-tune actually), try to unfreeze more parameters layer by layer and a high ratio of $\frac{LearningRate}{BatchSize}$(eg. 1e-4/64) makes the model generalize well.

## Data

To avoid overfitting, it's necessary to drop prompts having a over high cosine similarity(eg. 0.95).

Otherwise, more valuable data can lead to a significant lift.

## Discoveries

1. Noticing that the norms of models' outputs vary because cosine similarity loss does not constrain the norm, I find the prevalent ensembling method has a risk to be paralyzed by the inconsistency of norms and then correctify it by adding a scaling step which makes the result slightly better.

2. Data in this Competition has a characteristic that a prompt can match(generate actually) infinate images. I quantity this by comparing the text/image embeddings encoded by CLIP and then find that：

   a. Maybe there is a mismatch on this dimension of data between our train set and the dataset used to score on Kaggle, and then I try to narrow this gap by setting sample weight, which makes a 0.03+ score lift.

   b. I also find that a high-score open-source solution on Kaggle may enlarge this kind of gap by data analysis.

Details are in my discussions on Kaggle:

> https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/407264
>
> https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/409747
>
> https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/discussion/409763



