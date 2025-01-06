import sys
import string

class Vocab:
    def __init__(self, word_count_threshold: int = 10) -> None:
        self.word_count_threshold: int = word_count_threshold
        self.word_vocab: list[str] = []
        self.word_to_index: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.word_vocab)

class Helper:
    def __init__(self) -> None:
        pass

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

                if image_name not in new_captions_dict:
                    new_captions_dict[image_name] = []
                new_captions_dict[image_name].append(caption_compile)

        return new_captions_dict

    def build_vocab(self, processed_caption: dict[str, list[str]], word_count_threshold: int = 10) -> Vocab:
        word_count_dict: dict[str, int] = {}
        for _, captions_list in processed_caption.items():
            captions_list: list[str]
            for captions in captions_list:
                captions: str
                for word in captions.split():
                    word: str
                    if word not in word_count_dict:
                        word_count_dict[word] = 1
                    else: word_count_dict[word] += 1

        word_index: int = 0
        word_vocab: list[str] = []
        word_to_index: dict[str, int] = {}
        for word, count in word_count_dict.items():
            if count >= word_count_threshold:
                word_vocab.append(word)
                word_to_index[word] = word_index

        vocab = Vocab(word_count_threshold=word_count_threshold)
        vocab.word_vocab = word_vocab
        vocab.word_to_index = word_to_index

        return vocab




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
"""
