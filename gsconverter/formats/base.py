from abc import ABC, abstractmethod
import numpy as np

class BaseFormat(ABC):
    def __init__(self):
        self.extra_elements = []  # To store non-vertex PLY elements (PlyElement objects)
    @abstractmethod
    def read(self, path: str, **kwargs) -> np.ndarray:
        """
        Reads the file and returns a structured numpy array.
        
        Args:
            path (str): Path to the file.
            **kwargs: Additional arguments.
            
        Returns:
            np.ndarray: Structured numpy array containing the point cloud data.
        """
        pass

    @abstractmethod
    def write(self, data: np.ndarray, path: str, **kwargs) -> None:
        """
        Writes the structured numpy array to the file.
        
        Args:
            data (np.ndarray): Structured numpy array containing the point cloud data.
            path (str): Path to the output file.
            **kwargs: Additional arguments.
        """
        pass
