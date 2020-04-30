# How to create a covid dataset by retraining Stylegan2

This projects attemps to create a synthetic dataset for the purposes of training ML/DL  models to detect covid-19 and other respiratory diseases from x-rays.
A common problem in these cases is that there is a lack of positive cases that creates unbalanced datasets. By creating synthetic images we hope to alleviate this problem.

## Data used are from the following repository:

https://github.com/ieee8023/covid-chestxray-dataset

## Training

The implementation is based on the StyleGAN2 code provided by nvidia.

https://github.com/NVlabs/stylegan2



