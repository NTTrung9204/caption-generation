import sys
import string


class Helper:
    def __init__(self):
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
"""
