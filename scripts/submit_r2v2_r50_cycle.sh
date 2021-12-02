#!/bin/sh

pip install kornia --no-deps
pip install -v -e .

PRETRAIN=$1

python projects/siamfc-pytorch/train_siamfc.py \
  configs/representation/ssp/ssp_r50_nc_sgd_cos_100e_r5_1xNx2_k400.py \
  --pretrained $PRETRAIN --suffix cycle_r50_r2v2full_200ep --disable-wandb  \
  --moco-pretrain --workers 10
