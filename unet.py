import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
		)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        outputs = self.conv2(x)
        return outputs


class UNet(nn.Module):
    def __init__(self, n_class, bilinear=True):
        super(UNet, self).__init__()
        self.n_class = n_class
        self.bilinear = bilinear

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# down sampling
        self.conv1 = DoubleConv(3, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)

		# up sampling
        self.detrans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.deconv1  = DoubleConv(1024, 512)
        self.detrans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.deconv2  = DoubleConv(512, 256)
        self.detrans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.deconv3  = DoubleConv(256, 128)
        self.detrans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.deconv4  = DoubleConv(128, 64)
		
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0)
	
    def crop_image(self, x1, x2):
        x1_size = x1.size()[2]
        x2_size = x2.size()[2]
        delta = (x1_size - x2_size) // 2
        return x1[:, :, delta:x1_size-delta, delta:x1_size-delta]

    def forward(self, x):
        # Complete the forward function for the rest of the encoder

        x1 = self.conv1(x)
        x2 = self.max_pool(x1)
        x3 = self.conv2(x2)
        x4 = self.max_pool(x3)
        x5 = self.conv3(x4)
        x6 = self.max_pool(x5)
        x7 = self.conv4(x6)
        x8 = self.max_pool(x7)
        x9 = self.conv5(x8)

        # Complete the forward function for the rest of the decoder
        # score = self.classifier(out_decoder) 
        x = self.detrans1(x9)
#         y = self.crop_image(x7, x)
        x = self.deconv1(torch.cat([x,x7],1))

        x = self.detrans2(x)
#         y = self.crop_image(x5, x)
        x = self.deconv2(torch.cat([x,x5],1))

        x = self.detrans3(x)
#         y = self.crop_image(x3, x)
        x = self.deconv3(torch.cat([x,x3],1))

        x = self.detrans4(x)
#         y = self.crop_image(x1, x)
        x = self.deconv4(torch.cat([x,x1],1))

        score = self.classifier(x)

        # print(score.size())
        return score
	


if __name__=="__main__":
	image = torch.rand(1, 3, 256,128)
	model = UNet(27)
	model(image)


