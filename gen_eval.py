# needs to import this in the beginning to work on SLURM
import skimage
from skimage import filters
import ctypes

libgcc_s = ctypes.CDLL("libgcc_s.so.1")

import torch
import argparse
from utils.factories import dotdict
import utils.eval_pipeline


parser = argparse.ArgumentParser(
    description="Define hyperparameters.", prefix_chars="-"
)

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument(
    "--dataset", type=str, default="CelebA", help="CIFAR10, CIFAR100, RImgNet"
)
parser.add_argument(
    "--doc_id", type=int, required=True, help="doc_id of model within the database"
)

hps = parser.parse_args()


def main():
    assert hps.dataset in ["CIFAR10", "CIFAR100", "RImgNet", "CelebA"]
    utils.eval_pipeline.gen_eval(
        hps.doc_id, device=torch.device("cuda:" + str(hps.gpu))
    )


if __name__ == "__main__":
    main()
