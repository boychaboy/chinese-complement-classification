import argparse
import os

import nsml
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument("--mode", help="declare train or eval 
