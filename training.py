import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN
from torchvision.models.inception import Inception3, inception_v3



def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)), # 5atér inception model yé5ou input size (299,299)
            # donc hna héthi tétsama data augmentation 5atér fkol mara bch tjbd (299,299) mn (356,356)
            # donc kol mara bch tétjbéd taswira
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder=r"C:\Users\Asus\Desktop\Custom data\flickr8k\images", # path ta3 dossier li fih tsawér
        annotation_file=r"C:\Users\Asus\Desktop\Custom data\flickr8k\captions.txt", # path ta3 fichier text li fih ...
        transform=transform, # t3adi transform hna
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True # boosting performances of the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # tésta3ml GPU si valide si nn CPU
    load_model = True
    save_model = False
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    # writer = SummaryWriter("runs/flickr")
    # step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device) # création d'un modèle
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  # don't care about the pad index
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN


    # if load_model: # kén fama loading hétha
    #     step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    print_examples(model, device, dataset)

    model.train() # traja3 model ll train mod
    # for epoch in range(num_epochs):
    #
    #     if save_model: # kénk bch ta3ml save
    #         checkpoint = { # t7a4ar checkpoint él dictionnaire li bch ta3mlou save
    #             "state_dict": model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "step": step,
    #         }
    #         save_checkpoint(checkpoint)
    #
    #     for (imgs, captions) in tqdm( # forward prop fi epoch
    #             (train_loader), total=len(train_loader)
    #     ):
    #         print(type(imgs))
    #         captions = captions.to(device)
    #
    #         outputs = model(imgs, captions[:-1]) # bch n3adiw captions kol sauf lokhrania khatér n7ébou model mté3na
    #         # yét3alam ya3ml prédiciton ll <TN> word
    #         # outputs ... fi wostha predicted phrase
    #         loss = criterion( # calcul de loss
    #             outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1) # resizing mieux expliquer dans
    #             # machine translation
    #         )
    #
    #         # writer.add_scalar("Training loss", loss.item(), global_step=step)
    #         # step += 1
    #
    #         optimizer.zero_grad() # bachward propagation
    #         loss.backward(loss)
    #         optimizer.step()
    #

if __name__ == "__main__":
    train()

