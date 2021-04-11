import torch.nn as nn
from torchvision import models
import torch
# from torchsummary import summary

class Transfer(nn.Module):

    def __init__(self, n_class):
        super(Transfer, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        self.n_class = n_class
        self.encoder = vgg.features

        for params in self.encoder.parameters():
            params.requires_grad = False

        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class , kernel_size=1)

        torch.nn.init.xavier_uniform(self.deconv1.weight.data)
        torch.nn.init.xavier_uniform(self.deconv2.weight.data)
        torch.nn.init.xavier_uniform(self.deconv3.weight.data)
        torch.nn.init.xavier_uniform(self.deconv4.weight.data)
        torch.nn.init.xavier_uniform(self.deconv5.weight.data)
        torch.nn.init.xavier_uniform(self.classifier.weight.data)

    def forward(self, x):
        # x1 = __(self.relu(__(x)))
        # Complete the forward function for the rest of the encoder
        # score = __(self.relu(__(out_encoder)))
        x = self.encoder(x)


        # Complete the forward function for the rest of the decoder
        # score = self.classifier(out_decoder) 
        
        x = self.bn1(self.deconv1(x))
        x = self.relu(x) 
        x = self.bn2(self.deconv2(x))
        x = self.relu(x)
        x = self.bn3(self.deconv3(x))
        x = self.relu(x) 
        x = self.bn4(self.deconv4(x))
        x = self.relu(x) 
        x = self.bn5(self.deconv5(x))
        x = self.relu(x) 
        score = self.classifier(x)
                          

        return score  # size=(N, n_class, x.H/1, x.W/1)

if __name__=="__main__":
	# image = torch.rand(1, 3, 256,128)
    model = Transfer(n_class=27)
	# summary(model, image)
