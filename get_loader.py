import os
import pandas as pd 
import spacy  
import torch
from torch.nn.utils.rnn import pad_sequence  
from torch.utils.data import DataLoader, Dataset
from PIL import Image  
import torchvision.transforms as transforms


spacy = spacy.load("en_core_web_sm")


class vocabulaire:
    def __init__(self,threshold):
        self.UNK = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.PAD = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.threshold = threshold

    def __len__(self):
        return len(self.PAD)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequences = {}
        idices = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequences[word] = 1

                else:
                    frequences[word] += 1

                if frequences[word] == self.freq_threshold:
                    self.UNK[word] = idx
                    self.PAD[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.UNK[token] if token in self.UNK else self.UNK["<UNK>"]
            for token in tokenized_text
        ]


class Dataset(Dataset):
    def __init__(self, root_dir, captions_file, threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        caption = [self.vocab.stoi["<SOS>"]]
        caption += self.vocab.numericalize(caption)
        caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = Dataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.UNK["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
      
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        r"C:\Users\Asus\Desktop\Custom data\flickr8k\images", r"C:\Users\Asus\Desktop\Custom data\flickr8k\captions.txt", transform=transform
    )

    
