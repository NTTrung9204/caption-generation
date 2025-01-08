import sys
import string
import random
from keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.utils import to_categorical # type: ignore
import torch
import pickle

class Vocab:
    def __init__(self, word_count_threshold: int = 10) -> None:
        self.word_count_threshold: int = word_count_threshold
        self.word_vocab: list[str] = []
        self.word_to_index: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.word_vocab) + 1


class Helper:
    def __init__(self, device: str) -> None:
        self.device = device
    
    def save_features_to_file(self, features: dict, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {file_path}")

    def load_features_from_file(self, file_path: str) -> dict:
        with open(file_path, 'rb') as f:
            features = pickle.load(f)
        print(f"Features loaded from {file_path}")
        return features

    def clean_string(self, original_string: str) -> str:
        return (
            original_string.replace("\n", "").replace(".", "").replace('"', "").strip()
        )

    def load_captions(self, pathname: str) -> dict[str, list[str]]:
        captions_dict: dict = {}
        try:
            with open(pathname, "r") as file:
                content_file: list[str] = file.readlines()
                total_lines: int = len(content_file)
                for index_line, content_line in enumerate(content_file, 1):
                    index_line: int
                    content_line: str
                    if ".jpg" in content_line:
                        content_line_split: list[str] = content_line.split(",")
                        image_name: str = content_line_split[0]
                        desciption: str = self.clean_string(
                            "".join(content_line_split[1:])
                        )
                        if image_name not in captions_dict:
                            captions_dict[image_name] = []
                        captions_dict[image_name].append(desciption)
                        sys.stdout.write(
                            f"\rLoading text file, {index_line} | {total_lines}, image name: {image_name}"
                        )
        except FileNotFoundError:
            raise Exception("File not found!")
        except Exception as e:
            raise Exception(f"Error: {e}")

        print("\nTotal captions:", len(captions_dict))

        return captions_dict

    def captions_processing(
        self, captions_dict: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        new_captions_dict: dict[str, list[str]] = {}
        table: dict = str.maketrans("", "", string.punctuation)
        for image_name, captions_list in captions_dict.items():
            image_name: str
            captions_list: list[str]
            for caption in captions_list:
                caption: str
                words_list: list[str] = caption.split()
                words_list = [word.lower() for word in words_list]
                words_list = [word.translate(table) for word in words_list]
                words_list = [word for word in words_list if len(word) > 1]
                words_list = [word for word in words_list if word.isalpha()]
                caption_compile: str = " ".join(words_list)
                caption_compile = f"startseq {caption_compile} endseq"

                if image_name not in new_captions_dict:
                    new_captions_dict[image_name] = []
                new_captions_dict[image_name].append(caption_compile)

        return new_captions_dict

    def build_vocab(
        self, processed_caption: dict[str, list[str]], word_count_threshold: int = 10
    ) -> Vocab:
        word_count_dict: dict[str, int] = {}
        for _, captions_list in processed_caption.items():
            captions_list: list[str]
            for captions in captions_list:
                captions: str
                for word in captions.split():
                    word: str
                    if word not in word_count_dict:
                        word_count_dict[word] = 1
                    else:
                        word_count_dict[word] += 1

        word_index: int = 1
        word_vocab: list[str] = ['']
        word_to_index: dict[str, int] = {}
        for word, count in word_count_dict.items():
            if count >= word_count_threshold:
                word_vocab.append(word)
                word_to_index[word] = word_index
                word_index += 1

        vocab = Vocab(word_count_threshold=word_count_threshold)
        vocab.word_vocab = word_vocab
        vocab.word_to_index = word_to_index

        return vocab
    
    def calculate_max_length(self, data_set: dict[str, list[str]]):
        all_captions: list[str] = []
        for image_name in data_set.keys():
            [all_captions.append(caption) for caption in data_set[image_name]]
        return max(len(caption.split()) for caption in all_captions)

    def generate_dataset_structure(self, data_set: dict[str, list[str]], image_encoding: dict[str, torch.Tensor], vocab: Vocab) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        X1: list[torch.Tensor] = []
        X2: list[torch.Tensor] = []
        Y: list[torch.Tensor] = []
        max_length = self.calculate_max_length(data_set)
        print("Max length padding: ", max_length)
        for image_name, captions_list in data_set.items():
            image_name: str
            captions_list: list[str]
            for captions in captions_list:
                captions: str
                seq: list[str] = [vocab.word_to_index[word] for word in captions.split() if word in vocab.word_vocab]

                for index in range(1, len(seq)):
                    in_seq, out_seq = seq[:index], seq[index]

                    X1.append(torch.tensor(image_encoding[image_name]).to(self.device))
                    # print(in_seq)
                    X2.append(torch.tensor(pad_sequences([in_seq], maxlen=max_length)[0]).to(self.device))
                    # print(X2[-1])
                    # print(X2[-1].shape)
                    # n = input()
                    # print(out_seq)
                    # print(to_categorical([out_seq], num_classes=len(vocab))[0])
                    # print(list(to_categorical([out_seq], num_classes=len(vocab))[0]))
                    # n = input()
                    Y.append(torch.tensor(to_categorical([out_seq], num_classes=len(vocab))[0]).to(self.device))

        return X1, X2, Y

    def split_dataset(
        self, processed_caption: dict[str, list[str]], val_size: int, test_size: int
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        image_name_list: list[str] = list(processed_caption.keys())
        random.seed(43)
        random.shuffle(image_name_list)

        val_image_name_list: list[str] = image_name_list[:val_size]
        test_image_name_list: list[str] = image_name_list[val_size : val_size + test_size]
        train_image_name_list: list[str] = image_name_list[val_size + test_size:]

        val_set: dict[str, list[str]] = {image_name: processed_caption[image_name] for image_name in val_image_name_list}
        test_set: dict[str, list[str]] = {image_name: processed_caption[image_name] for image_name in test_image_name_list}
        train_set: dict[str, list[str]] = {image_name: processed_caption[image_name] for image_name in train_image_name_list}

        return train_set, val_set, test_set



"""
    [TEST]
    pathname = "dataset/captions.txt"

    captions = Helper().load_captions(pathname) -> 8091 image name

    print(captions["1012212859_01547e3f17.jpg"])

    [RESULT]
    [
        "A dog shakes its head near the shore  a red ball next to it",
        "A white dog shakes on the edge of a beach with an orange ball",
        "Dog with orange ball at feet  stands on shore shaking off water",
        "White dog playing with a red ball on the shore near the water",
        "White dog with brown ears standing near water with head turned to one side",
    ]

    [TEST]
    processed_caption = Helper().captions_processing(captions)

    print(processed_caption["1012212859_01547e3f17.jpg"])

    [RESULT]
    [
        "dog shakes its head near the shore red ball next to it",
        "white dog shakes on the edge of beach with an orange ball",
        "dog with orange ball at feet stands on shore shaking off water",
        "white dog playing with red ball on the shore near the water",
        "white dog with brown ears standing near water with head turned to one side",
    ]

    [TEST]
    vocab = Helper().build_vocab(processed_caption, word_count_threshold=10)

    print(len(vocab))

    [RESULT]
    1839

    [TEST]
    dataset_size = Helper().get_dataset_size(processed_caption, vocab)

    print(dataset_size)

    val_size = int(len(processed_caption) * 0.1)
    test_size = int(len(processed_caption) * 0.1)

    print(val_size, test_size)

    train_set, val_set, test_set = Helper().split_dataset(processed_caption, val_size, test_size)
    print(Helper().generate_dataset_structure(train_set, vocab))
    print(Helper().generate_dataset_structure(val_set, vocab))
    print(Helper().generate_dataset_structure(test_set, vocab))

    [RESULT]
    val_size_image: 809, test_size_image: 809
    train_set_size: 285390
    val_set_size: 35371
    test_set_size: 35833
"""

# pathname = "dataset/captions.txt"

# captions = Helper().load_captions(pathname)

# processed_caption = Helper().captions_processing(captions)

# vocab = Helper().build_vocab(processed_caption, word_count_threshold=10)

# print(Helper().calculate_max_length(processed_caption))

# X1, X2, Y = Helper().generate_dataset_structure(processed_caption, vocab)

# print(Y)

