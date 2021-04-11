import matplotlib.pyplot as plt 
import numpy as np 
from dataloader import * 

def plotLoss(train_loss, valid_loss): 
    #plot the figure of training and validation loss 
    plt.plot(train_loss, label='Training loss')
    plt.plot(valid_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.savefig('3-BaseLine-Loss.png')
    plt.close()


def plotAcc(train_acc, valid_acc): 
    #plot the figure of training and validation accuracy 
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(valid_acc, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation accuracy per Epoch')
    plt.legend()
    plt.savefig('3-BaseLine-Accuracy.png')
    plt.close()

def visualizeImage(inputImg):
    # inputImage n_class * h * w 
    # outputImage h * w * 3 
    
    img = np.array(inputImg)
    #print(img.shape)
    predictImg = np.argmax(img, 0)
    
    outputImage = np.zeros((img.shape[1], img.shape[2], 3))
    
    
    for h in range(predictImg.shape[0]):
        for w in range(predictImg.shape[1]):
            color = labels[predictImg[h][w]][2]
            outputImage[h][w][0] = color[0] / 255
            outputImage[h][w][1] = color[1] / 255
            outputImage[h][w][2] = color[2] / 255
            
    plt.imshow(outputImage)
    plt.title('Visualization of segmented output')
    plt.savefig('3-Baseline-visualizeTest.png')
    plt.close()
     