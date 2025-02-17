from .CNN1d import CNNRULRegression
from .CNN2d import CNNRUL2DRegression
from .hybridCNN import HybridCNNRegression, ComplexHybridCNNRegression
from .TCN import TCNRegression

__all__ = ["CNNRULRegression", "CNNRUL2DRegression", "HybridCNNRegression", "ComplexHybridCNNRegression",
           "TCNRegression"]
