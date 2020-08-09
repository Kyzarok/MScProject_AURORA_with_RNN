import torch
import torch.nn as nn
import torch.nn.functional as F


# Autoencoder come in two parts, the encoder and the decoder
class Encoder(nn.Module):
    def __init__(self, conf = [5], enc_dim = 2):

        # conv  = Conv2d(input, input size, in channels, outchannels, filter size)
        #               input,  (2,50), 1, 2, (2,6)
        super(Encoder, self).__init__()
        # self.aurora_shape = [-1, 2, 50, 1]
        self.conv2d = nn.Conv2d(1, 2, (2, 6))
        # fc = fc( input, n_in, n_out) 
        if len(conf) == 1:
            self.fc1 = nn.Linear(2, conf[0])
            self.fc2 = nn.Linear(conf[0], enc_dim)
    
    def forward(self, x):
        out = self.conv2d(x)
        out = F.leaky_relu(out)
        out = F.max_pool2d(out, (2,2))
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.sigmoid(out)
        out = self.fc2(out)
        return out

class Decoder(nn.Module):
    def __init__(self,  conf = [5, 50*2], enc_dim = 2, output_dim = 100):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2, conf[0])
        self.fc2 = nn.Linear(conf[0], conf[1])

        #input, output_siz, in_ch, out_ch, patch_siz, activation='relu'
        #self.layers[-1],(self.layers[-1].get_shape().as_list()[1], self.layers[-1].get_shape().as_list()[2]), self.layers[-1].get_shape().as_list()[3], conf[i],
                                     #(2, 6), activation='leak_relu')
        self.conv2d_t1 = nn.ConvTranspose2d(24, 2, (2, 6))
        self.conv2d_t2 = nn.ConvTranspose2d(14, 1, (2, 6))

        self.fc3 = nn.Linear(30, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.sigmoid(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        out = self.conv2d_t1(out)
        out = F.leaky_relu(out)
        out = self.conv2d_t2(out)
        out = F.leaky_relu(out)
        out = self.fc3(out)
        return out
