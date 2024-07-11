import sys
from torchvision.datasets import MNIST
assert len(sys.argv) > 1
MNIST(sys.argv[1], train=True, download=True)

