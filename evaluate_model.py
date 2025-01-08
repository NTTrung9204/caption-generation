import torch
from build_model.inception_v3_model import InceptionV3Model
from build_model.model import CaptionGenerationNet
from helper import Helper
from prepare_data import SeqDataset
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import sys
import psutil
import matplotlib.pyplot as plt

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for datas, labels in data_loader:
            labels = labels.to(device)

            outputs = model(datas)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            labels = labels.argmax(dim=1)
            correct += (predicted == labels).sum().item()

            sys.stdout.write(f"\rTotal: {total} | Correct: {correct}")

    return correct / total * 100


if __name__ == "__main__":
    process = psutil.Process()

    device = torch.device("cuda")
    print(device)

    captions_pathname = "dataset/captions.txt"
    images_pathname = "dataset/images/"
    encoding_images_dict_path = "encoding_images_dict.pkl"

    helper_func = Helper(device)

    captions = helper_func.load_captions(captions_pathname)

    processed_caption = helper_func.captions_processing(captions)

    vocab = helper_func.build_vocab(processed_caption, word_count_threshold=10)
    vocab_size = len(vocab)

    val_size = int(len(processed_caption) * 0.3)
    test_size = int(len(processed_caption) * 0.1)

    train_set, val_set, test_set = helper_func.split_dataset(
        processed_caption, val_size, test_size
    )

    feature_extractor_model = InceptionV3Model(device)

    model = torch.load("trained_model_v5.pth").to(device)
    model.eval()

    encoding_images_dict = helper_func.load_features_from_file(
        encoding_images_dict_path
    )

    print("Length of images encoding:", len(encoding_images_dict))

    x_1_val_set, x_2_val_set, y_val_set = helper_func.generate_dataset_structure(
        val_set, encoding_images_dict, vocab
    )

    print(f"Val set size: {len(x_1_val_set)}")

    val_dataset = SeqDataset(x_1_val_set, x_2_val_set, y_val_set)

    val_set_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)

    print(f"Accucary: {evaluate_accuracy(model, val_set_loader, device)}")
