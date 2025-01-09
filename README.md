
# Project: Image Captioning using Pretrained Model
## Description:
This script uses a pretrained model to generate captions for a set of test images. It extracts features using an InceptionV3 model and applies a Caption Generation Network (CGN) to generate descriptive captions.

## Architecture.
There are two fundamental approaches to inputting for image captioning: Injection Architecture and Merging Architecture:
![image](https://github.com/user-attachments/assets/0ac245aa-8495-426a-bdcf-d4fefe43f4e1)

Merging Architecture is used here:
![image](https://github.com/user-attachments/assets/bf483e1d-8c94-4058-b4bb-9ad493eaa061)

## CNN Model.
Utilizing the InceptionV3 model to extract image features, resulting in a 2048-dimensional vector.
![image](https://github.com/user-attachments/assets/4fde3b93-ab0b-4e86-afe7-b12f82899803)

## RNN Model.
Using skip-gram for text encoding.
![image](https://github.com/user-attachments/assets/090e4e4b-b227-4ffa-b306-3a905c9f57f9)

Using LSTM to extract word information.
![image](https://github.com/user-attachments/assets/cc0f62e0-e584-4987-96f6-7ab3d01af914)

## Training
![training_loss](https://github.com/user-attachments/assets/129d9f3e-5b1e-4a11-bd6f-c4a06f7a6a7b)

## Result
![test_model](https://github.com/user-attachments/assets/4ed215ea-f858-4ffc-ad80-84524ccac4f9)

## Contact

If you have any questions, please contact me at: [trung9204@gmail.com].

## License

This project is licensed under the [MIT License](LICENSE).
