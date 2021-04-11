from torchvision import utils
import torchvision.transforms as transforms
from basic_fcn import *
from dataloader import *
from utils import *
from plot import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

## 
epochs = 30
num_batch_size= 16
## 
# TODO: Some missing values are represented by '__'. You need to fill these up.
print("Load data...")

train_dataset = IddDataset(csv_file='train.csv')
#train_dataset_aug = IddDataset(csv_file='train.csv', aug = True)
#train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_aug])                                
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')


train_loader = DataLoader(dataset=train_dataset, batch_size= num_batch_size, num_workers= 0, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= num_batch_size, num_workers= 0, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= num_batch_size, num_workers= 0, shuffle=False)
print("initial weight...")
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
#         torch.nn.init.xavier_uniform(m.bias.data)
        m.bias.data.fill_(0.01)

total = torch.zeros([27], dtype=torch.float)

for iter, (_, _, Y) in enumerate(train_loader): 
    area = torch.histc(Y.float().cpu(), bins=27, min=0, max=26)
    total += area

normedWeights = total

normedWeights = 1 / normedWeights

normedWeights = torch.sqrt(normedWeights)
normedWeights = torch.sqrt(normedWeights)
normedWeights = torch.sqrt(normedWeights)
normedWeights[26] = min(normedWeights)          # set unlabeled class less weight
normedWeights = normedWeights / normedWeights.sum()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
        
criterion = nn.CrossEntropyLoss(weight=normedWeights.cuda()).cuda()
fcn_model = FCN(n_class=27)
fcn_model.apply(init_weights)


optimizer = optim.Adam(fcn_model.parameters(), lr=1e-2)

use_gpu = torch.cuda.is_available()

if use_gpu:
    fcn_model = fcn_model.cuda()


def train():
    stopping_epoch = 3
    min_val_loss = 1000

    trainLossEpoch = []
    trainAccEpoch = [] 
    trainIOUEpoch = []
    validLossEpoch = []
    validAccEpoch = [] 
    validIOUEpoch = []
    
    for epoch in range(epochs):
#         print("epoch:",epoch)
        ts = time.time()
        loss_mini_batch = []
        accuracy_mini_batch = []
        iou_mini_batch = []
        
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()
                labels = tar.cuda()
                Y = Y.cuda()
            else:
                inputs, labels = X, tar
            outputs = fcn_model(inputs)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            probas = F.softmax(outputs, dim=1)
            accuracy_ = pixel_acc(probas, labels)
            ious, avg_iou = iou(probas, labels)

            loss_mini_batch.append(loss.item())
            accuracy_mini_batch.append(accuracy_)
            iou_mini_batch.append(avg_iou)

            if iter % 50 == 0:
                print("Training: epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
                
        train_loss = sum(loss_mini_batch)/len(loss_mini_batch)
        train_accuracy = sum(accuracy_mini_batch)/len(accuracy_mini_batch)
        train_iou = sum(iou_mini_batch)/len(iou_mini_batch)
        print("Finish training epoch {}, time elapsed {}, iou: {}, accuracy: {}, loss: {}".format(epoch, time.time() - ts, train_iou, train_accuracy, train_loss))

        trainLossEpoch.append(train_loss)
        trainAccEpoch.append(train_accuracy) 
        trainIOUEpoch.append(train_iou)
        
        val_loss, val_acc, val_iou = val(epoch)
        
        validLossEpoch.append(val_loss)
        validAccEpoch.append(val_acc) 
        validIOUEpoch.append(val_iou)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(fcn_model, 'best_model')
            stopping_epoch = 3
        else:
            stopping_epoch = stopping_epoch - 1

        if stopping_epoch == 0:
            break
        

        fcn_model.train()
        
    plotLoss(trainLossEpoch, validLossEpoch)
    plotAcc(trainAccEpoch, validAccEpoch)

def val(epoch):
    ts = time.time()
    fcn_model.eval() # Don't forget to put in eval mode !
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    loss_mini_batch = []
    accuracy_mini_batch = []
    iou_mini_batch = []
    with torch.no_grad():
        for iter, (X, tar, Y) in enumerate(val_loader):

            if use_gpu:
                inputs = X.cuda()
                labels = tar.cuda()
                Y = Y.cuda()
            else:
                inputs, labels = X.to(torch.device('cpu')), tar.to(torch.device('cpu'))
            outputs = fcn_model(inputs)
            loss_ = criterion(outputs, Y)
            probas = F.softmax(outputs, dim=1)
            accuracy_ = pixel_acc(probas, labels)
            ious, avg_iou = iou(probas, labels)
            loss_mini_batch.append(loss_.item())

            accuracy_mini_batch.append(accuracy_)
            iou_mini_batch.append(avg_iou)
    
    loss = sum(loss_mini_batch)/len(loss_mini_batch)
    accuracy = sum(accuracy_mini_batch)/len(accuracy_mini_batch)
    IOU = sum(iou_mini_batch)/len(iou_mini_batch)
    print("Validation: epoch{}, iou: {}, accuracy: {}, loss: {}".format(epoch, IOU, accuracy, loss))
    print("ious for classes 0, 2, 9, 17, 25 {}".format(ious))
    print("time for validation: {}".format(time.time() - ts))
    return loss, accuracy, IOU         
    
def test():
    print("loading trained model")
    fcn_model = torch.load('best_model')
    fcn_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    loss_mini_batch = []
    accuracy_mini_batch = []
    iou_mini_batch = []
    with torch.no_grad():
        for iter, (X, tar, Y) in enumerate(test_loader):

            if use_gpu:
                inputs = X.cuda()
                labels = tar.cuda()
                Y = Y.cuda()
            else:
                inputs, labels = X.to(torch.device('cpu')), tar.to(torch.device('cpu'))
            outputs = fcn_model(inputs)
            probas = F.softmax(outputs, dim=1)
            if iter == 0: 
                visualizeImage(probas[0].cpu())
                #visualizeImage(tar[0].cpu())
            loss_ = criterion(outputs, Y)
            
            accuracy_ = pixel_acc(probas, labels)
            ious, avg_iou = iou(probas, labels)
            loss_mini_batch.append(loss_.item())

            accuracy_mini_batch.append(accuracy_)
            iou_mini_batch.append(avg_iou)
    
    loss = sum(loss_mini_batch)/len(loss_mini_batch)
    accuracy = sum(accuracy_mini_batch)/len(accuracy_mini_batch)
    IOU = sum(iou_mini_batch)/len(iou_mini_batch)

    print("Testing: iou: {}, accuracy: {}, loss: {}".format(IOU, accuracy, loss))
    print("ious for id 0, 2, 9, 17, 25 {}:".format(ious))
    
if __name__ == "__main__":
    print("show the accuracy before training")
    val(0)  # show the accuracy before training
    print("================start training===============")
    train()
    print("================start testing===============")
    test()
