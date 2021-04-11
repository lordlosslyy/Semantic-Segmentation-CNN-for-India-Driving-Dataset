import torch

def iou(pred, target):
    max_idx = torch.argmax(pred, dim=1, keepdim=True)
    one_hot = torch.cuda.FloatTensor(pred.shape)
    one_hot.zero_()
    one_hot.scatter_(dim=1, index=max_idx, value=1.)

    TP = torch.sum((one_hot[:, :-1, :, :] == target[:, :-1, :, :]) * (target[:, :-1, :, :] == 1)).item()
    FN_FP = torch.sum(one_hot[:, :-1, :, :] != target[:, :-1, :, :]).item() - torch.sum(one_hot[:, -1, :, :] != target[:, -1, :, :]).item()
    avg_iou = TP / (TP + FN_FP)

    ious = []
    for cls in [0, 2, 9, 17, 25]:
    # for cls in [0, 1, 2]: # for test
        labeled = target[:, -1, :, :] != 1
        TP = torch.sum((one_hot[:, cls, :, :] == target[:, cls, :, :]) * (target[:, cls, :, :] == 1)).item()
        FN_FP = torch.sum((one_hot[:, cls, :, :] != target[:, cls, :, :]) * labeled).item()
        if (TP + FN_FP) == 0:
            ious.append(float('nan'))
        else:
            ious.append(TP / (TP + FN_FP))

    return ious, avg_iou


def pixel_acc(pred, target):
    # pred 4D, target 4D
    max_idx = torch.argmax(pred, dim=1, keepdim=True)
    one_hot = torch.zeros_like(pred).scatter_(dim=1, index=max_idx, value=1.)
    pixel_labeled = torch.sum(target[:, :-1, :, :] == 1).item()
    pixel_correct = torch.sum((one_hot[:, :-1, :, :] == target[:, :-1, :, :]) * (target[:, :-1, :, :] == 1)).item()

    return pixel_correct / pixel_labeled

# test
# torch.Size([2, 4, 2, 3])
# test_target = torch.tensor([[[[1., 1., 1.],
#           [1., 1., 1.]],

#          [[0., 0., 0.],
#           [0., 0., 0.]],

#          [[0., 0., 0.],
#           [0., 0., 0.]],

#          [[0., 0., 0.],
#           [0., 0., 0.]]],


#         [[[0., 0., 0.],
#           [0., 0., 0.]],

#          [[0., 0., 0.],
#           [0., 0., 0.]],

#          [[0., 0., 0.],
#           [1., 1., 1.]],

#          [[1., 1., 1.],
#           [0., 0., 0.]]]])

# test_pred = torch.tensor([[[[.8, .8, .8],
#           [.8, .8, .8]],

#          [[.2, .2, .2],
#           [.2, .2, .2]],

#          [[0., 0., 0.],
#           [0., 0., 0.]],

#          [[0., 0., 0.],
#           [0., 0., 0.]]],


#         [[[0., 0., 0.],
#           [0., 0., 0.]],

#         [[.2, .8, .2],
#           [.8, .2, .8]],

#         [[.8, .2, .8],
#           [.2, .8, .2]],

#          [[0., 0., 0.],
#           [0., 0., 0.]]]])

# print(pixel_acc(test_pred, test_target), iou(test_pred, test_target))