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

# Định nghĩa hàm evaluate accuracy
def evaluate_accuracy(model, data_loader, device):
    model.eval()  # Chuyển sang chế độ đánh giá
    correct = 0
    total = 0

    with torch.no_grad():  # Không tính gradient
        for images, labels in data_loader:
            labels = labels.to(device)

            # Lấy dự đoán của mô hình
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)  # Lấy nhãn dự đoán (chỉ số max)

            # Cập nhật số lượng đúng và tổng số
            total += labels.size(0)
            labels = labels.argmax(dim=1)
            correct += (predicted == labels).sum().item()

    return correct / total * 100  # Trả về accuracy (%)

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

    train_set, val_set, test_set = helper_func.split_dataset(processed_caption, val_size, test_size)

    feature_extractor_model = InceptionV3Model(device)
    model = CaptionGenerationNet(vocab_size).to(device) 

    # image_name_path_list = [images_pathname + image_name for image_name in processed_caption.keys()]
    # print("Length of image list", len(image_name_path_list))

    # encoding_images_dict = feature_extractor_model.extract_features(image_name_path_list)

    encoding_images_dict = helper_func.load_features_from_file(encoding_images_dict_path)

    print("Length of images encoding:", len(encoding_images_dict))

    x_1_train_set, x_2_train_set, y_train_set = helper_func.generate_dataset_structure(train_set, encoding_images_dict, vocab)
    # x_1_val_set, x_2_val_set, y_val_set = helper_func.generate_dataset_structure(val_set, encoding_images_dict, vocab)

    print(f"Train set size: {len(x_1_train_set)}")
    # print(f"Val set size: {len(x_1_val_set)}")

    train_dataset = SeqDataset(x_1_train_set, x_2_train_set, y_train_set)
    # val_dataset = SeqDataset(x_1_val_set, x_2_val_set, y_val_set)

    # print(val_dataset.X1[0])
    # print(val_dataset.X2[0])
    # print(torch.sum(val_dataset.Y[0]))

    # print(val_dataset[0])

    train_set_loader = DataLoader(train_dataset, shuffle=True, batch_size=1024)

    print("Start training")

    memory_usage = process.memory_info().rss
    print(f"Memory: {memory_usage / (1024 * 1024):.2f} MB")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25
    losses = []
    # for epoch in range(num_epochs):
    #     for image, label in train_set_loader:
    #         outputs = model(image)
    #         loss = criterion(outputs, label.to(device))
    #         losses.append(loss.item())

    #         model.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         sys.stdout.write(f"\r{epoch}:{losses[-1]:.4f}")
    #     print()

    for epoch in range(num_epochs):
        model.train()  # Chuyển sang chế độ huấn luyện
        epoch_loss = 0.0

        for images, labels in train_set_loader:
            labels = labels.to(device)
            
            # Forward pass và tính loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Backpropagation và tối ưu hóa
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Tính loss trung bình trong epoch
        avg_loss = epoch_loss / len(train_set_loader)

        # Đánh giá sau mỗi epoch
        train_accuracy = evaluate_accuracy(model, train_set_loader, device)

        # Nếu có tập kiểm tra (validation), đánh giá thêm
        # val_accuracy = evaluate_accuracy(model, val_set_loader, device)

        # Hiển thị thông tin sau mỗi epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        # print(f"Validation Accuracy: {val_accuracy:.2f}%")  # Nếu có tập kiểm tra

    # plt.plot(losses)
    # plt.show()

    torch.save(model, 'trained_model_v3.pth')

