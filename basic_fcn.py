import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
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

    def forward(self, x):
        # x1 = __(self.relu(__(x)))
        # Complete the forward function for the rest of the encoder
        # score = __(self.relu(__(out_encoder)))
        
        x = self.bnd1(self.conv1(x))
        x = self.relu(x)
        x = self.bnd2(self.conv2(x))
        x = self.relu(x)
        x = self.bnd3(self.conv3(x))
        x = self.relu(x)
        x = self.bnd4(self.conv4(x))
        x = self.relu(x)
        x = self.bnd5(self.conv5(x))
        x = self.relu(x)


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