from torch.utils.data import Dataset
import numpy as np
import torch
from build_model.inception_v3_model import InceptionV3Model
from helper import Helper

class SeqDataset(Dataset):
    def __init__(self, X1: list[np.ndarray[np.float64]], X2: list[np.ndarray[np.int64]], Y: list[np.ndarray[np.int64]]) -> None:
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        # print("================================")
        # # print(self.X1)
        # print(X1[0])
        # print("================================")
        # print("================================")
        # # print(self.X1)
        # print(X1[1])
        # print("================================")
        # print("================================")
        # # print(self.X1)
        # print(X1[2])
        # print("================================")
        # print("================================")
        # # print(self.X1)
        # print(X1[3])
        # print("================================")

    def __len__(self) -> int:
        return len(self.X1)

    def __getitem__(self, index) -> tuple[tuple[np.ndarray[np.float64], np.ndarray[np.int64]], np.ndarray[np.int64]]:
        # print("index:", index)
        # print("Value: ", self.X1[index])
        # print("============================================")
        # print((self.X1[index], self.X2[index]), self.Y[index])
        return (self.X1[index], self.X2[index]), self.Y[index]
    

if __name__ == "__main__":
    device = torch.device("cuda")
    print(device)

    images_pathname = "dataset/images/"
    captions_pathname = "dataset/captions.txt"

    helper_func = Helper(device)

    captions = helper_func.load_captions(captions_pathname)

    processed_caption = helper_func.captions_processing(captions)

    feature_extractor_model = InceptionV3Model(device)

    image_name_path_list = [images_pathname + image_name for image_name in processed_caption.keys()]
    print("Length of image list", len(image_name_path_list))

    encoding_images_dict = feature_extractor_model.extract_features(image_name_path_list)

    helper_func.save_features_to_file(encoding_images_dict, "encoding_images_dict.pkl")