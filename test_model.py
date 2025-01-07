from build_model.model import CaptionGenerationNet
from build_model.inception_v3_model import InceptionV3Model
from helper import Helper
from keras.preprocessing.sequence import pad_sequences # type: ignore
import torch
import random
import numpy as np

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)

if __name__ == "__main__":
    # image_name = "58363928_6f7074608c.jpg"
    # image_name = "69189650_6687da7280.jpg"
    # image_name = "667626_18933d713e.jpg"
    # image_name = "3000722396_1ae2e976c2.jpg"
    # image_name = "300148649_72f7f0399c.jpg"
    image_name = "3004823335_9b82cbd8a7.jpg"

    encoding_images_dict_path = "encoding_images_dict.pkl"
    captions_pathname = "dataset/captions.txt"

    device = torch.device("cuda")

    helper_func = Helper(device)

    captions = helper_func.load_captions(captions_pathname)

    processed_caption = helper_func.captions_processing(captions)

    vocab = helper_func.build_vocab(processed_caption, word_count_threshold=10)
    vocab_size = len(vocab)

    model = torch.load("trained_model_v3.pth").to(device)
    model.eval()

    encoding_images_dict = helper_func.load_features_from_file(encoding_images_dict_path)

    image_input_encoding = torch.tensor(encoding_images_dict[image_name]).to(device)

    input_sequences = "startseq"

    in_seq = [vocab.word_to_index[word] for word in input_sequences.split()]

    max_length = helper_func.calculate_max_length(processed_caption)

    print("Max length:", max_length)

    seq_input_encoding = torch.tensor(pad_sequences([in_seq], maxlen=max_length)[0]).to(device)

    while True:
        if input_sequences.split()[-1] == "endseq" or len(input_sequences.split()) > max_length:
            print(input_sequences)
            break

        with torch.no_grad():
            output = model((image_input_encoding, seq_input_encoding))
            input_sequences += " "  + vocab.word_vocab[torch.argmax(output).cpu().detach().numpy()]
            in_seq = [vocab.word_to_index[word] for word in input_sequences.split()]

            seq_input_encoding = torch.tensor(pad_sequences([in_seq], maxlen=29)[0]).to(device)



