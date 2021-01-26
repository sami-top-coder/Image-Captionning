import torch
import torch.nn as nn
import statistics
import torchvision.models as models

class Encodeur(nn.Module): # Class bch yésna3 biha EncoderCNN kil 3ada elle doit hériter ml nn.Module
    def __init__(self, embed_size, train_CNN=False): # redéfinition du constructeur
        super(EncoderCNN, self).__init__() # appel ll constructeur de la classe mère
        self.train_CNN = train_CNN # bch t9olk bch na3mlou train ll CNN encoder wla .. dans notre cas la
        # bch nhabtou pretrained model 7a4ér w na3mloulou fine tunning ma3néha juste freeze the last layers
        self.inception = models.inception_v3(pretrained=True)   # bch nésta3mlou inception_v3
        # ka CNN encoder .. aux_logtis : tafi thaw 7aja téb3a InceptionV3
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        # bch naccédiw ll last fc layer ta3 inception layer w nbadlouha .. input howa inception.fc.in_features
        # output : embed_size bch najmou n3adiwha ll embeded word fl RNN .. (voir model in this project + RQ )
        self.relu = nn.ReLU() # fonction Relu ll model ...
        self.dropout = nn.Dropout(0.5) # Drop out regularization to avoid overfitiing

    def forward(self, images):  # redéfinition ll forward propagation# #

        features= self.inception(images)

        return self.dropout(self.relu(features[0]))

class Decodeur(nn.Module): # class ta3 décodeur
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers): # redéfinition ll constructeur kil 3ada
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # embedding layer : to map our word to some dimentional
        # space to have a better representation of the word
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        # num_layers : 9adéh mn LSTM stacked fou9 b3a4ha
        self.linear = nn.Linear(hidden_size, vocab_size)
        # linear layer : yé5ou output size ml LSTM li howa hidden_size w y5araj prédiction mté3o
        # Rq : one node == one vocab word
        self.dropout = nn.Dropout(0.5) # kél 3ada ll overfit

    def forward(self, features, captions): # features li 5arajhom el CNN , captions: houma el target
        embeddings = torch.cat((features.unsqueeze(0), embeddings) , dim=0)
        # + embedding words thothom fl embeddings donc embedding = features + captions
        hiddens, _ = self.lstm(embeddings)  # n3adiwha 3al lstm w né54ou juste hiddens
        outputs = self.linear(hiddens)  # w hna n3adiw hiddens 3al linear layer
        return outputs

class Encodeur_Decodeur(nn.Module): # kil3ada class CNNtoRNN hérite ml nn.Module
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoder = Encodeur(embed_size) # tésna3 EncoderCNN bl class loula
        self.decoder = Decodeur(embed_size, hidden_size, vocab_size, num_layers) # tésna3 DecoderRNN bl class
        # thénia

    def forward(self, images, captions):
        features = self.encoder(images) # t3adi images à travers encoderCNN bch t5araj features
        outputs = self.decoder(features, captions) # features + captions t3adihom à travers decoderRNN bch
        # tét7asal 3al outputs
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = [] # lista fiha les indices ta3 les mots li 3mélhom prédiction

        with torch.no_grad(): # sans calcul de gradients
            x = self.encoder(image).unsqueeze(0) # yhot  feature vector fi tenseur (1,n)
            # Rq : kén 3mélna unsqueeze(1) yhotha fi tenseur (n,1)
            etats = none

            for _ in range(max_length): # max_length : taille ta3 akbar jomla tjm to5roj (ta3mlha prédiction)
                caches , etats = self.decodeur.lstm(x, etats) # passage à travers states
                output = self.decodeur.linear(caches.squeeze(0)) # passage à travers linear bch tét7asal al output
                # output : hia matrice de probabilité
                prediction = output.argmax(1)  # hna n5arjou l'indice ta3 akbar probabilité .. li howa bch ykoun
                # index ta3 klma
                result_caption.append(prediction.item()) # nzidou index fil liste result_caption
                x = self.decodeur.embed(prediction).unsqueeze(0) #conversion du predicted word l embeded word

                if vocabulary.UNK[predicted.item()] == "<EOS>": # kén predicted word hia end of text
                    break # ta9sa

        return [vocabulary.itos[idx] for idx in result_caption] # ndouro 3al les indices kol bch ncovertiwhom lklmét
    # bch ywaliw jomla
