from build_model.model import CaptionGenerationNet
from build_model.inception_v3_model import InceptionV3Model
from helper import Helper
from keras.preprocessing.sequence import pad_sequences  # type: ignore
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda")

    helper_func = Helper(device)
    extractor = InceptionV3Model(device)

    captions_pathname = "dataset/captions.txt"
    encoding_images_dict_path = "encoding_images_dict.pkl"

    captions = helper_func.load_captions(captions_pathname)
    processed_caption = helper_func.captions_processing(captions)
    vocab = helper_func.build_vocab(processed_caption, word_count_threshold=10)
    vocab_size = len(vocab)

    model = torch.load("trained_model_v5.pth").to(device)
    model.eval()

    max_length = helper_func.calculate_max_length(processed_caption)

    image_files = [f"actual_test/test_{i}.jpg" for i in range(1, 6)]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, image_name in enumerate(image_files):
        image_input_encoding = torch.tensor(extractor.extract_features([image_name])[image_name.split("/")[-1]]).to(device)

        input_sequences = "startseq"
        in_seq = [vocab.word_to_index[word] for word in input_sequences.split()]
        seq_input_encoding = torch.tensor(pad_sequences([in_seq], maxlen=max_length)[0]).to(device)

        while True:
            if (
                input_sequences.split()[-1] == "endseq"
                or len(input_sequences.split()) > max_length
            ):
                break

            with torch.no_grad():
                output = model((image_input_encoding, seq_input_encoding))
                next_word = vocab.word_vocab[torch.argmax(output).cpu().detach().numpy()]
                input_sequences += " " + next_word
                in_seq = [vocab.word_to_index[word] for word in input_sequences.split()]
                seq_input_encoding = torch.tensor(pad_sequences([in_seq], maxlen=max_length)[0]).to(device)

        caption = input_sequences.replace("startseq", "").replace("endseq", "").strip()

        image = plt.imread(image_name)
        axes[i].imshow(image)
        axes[i].set_title(caption, fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
