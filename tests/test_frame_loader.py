import cv2
import matplotlib.pyplot as plt
from io_utils.dataset import dataset_factory
from config import Config

def test_frame_loader(img_id: int):
    config = Config()
    dataset = dataset_factory(config)
    if dataset.isOk():
        img = dataset.getImage(img_id)
        depth = dataset.getDepth(img_id)
        print(f"Loaded frame {img_id}: image shape {img.shape} and depth shape {depth.shape}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {img_id}")
        plt.axis('off')
        plt.show()
    else:
        print("Dataset not OK.")

if __name__ == "__main__":
    test_frame_loader(12)
