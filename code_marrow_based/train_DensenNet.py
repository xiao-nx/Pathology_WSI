from __future__ import print_function, division
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import time
import os
import copy
from tensorboardX import SummaryWriter
from sklearn import metrics
import xlwt
import xlrd
import pandas as pd

MODEL_PATH = './models_marrow_based/'

# 重写Dataset
class LoadMyDataset(torch.utils.data.Dataset):
    def __init__(self, txtdata, transform=None):
        with open(txtdata, 'r') as f:
            imgs, all_label = [], []
            for line in f.readlines():
                line = line.strip().split(" ")
                imgs.append((line[0], line[1]))
                if line[1] not in all_label:
                    all_label.append(line[1])
            classes = set(all_label)
            #print("classe number: {}".format(len(classes)))
            classes = sorted(list(classes))
            class_to_idx = {classes[i]: i for i in range(len(classes))}  # convert label to index(from 0 to num_class-1)
            del all_label

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        label = self.class_to_idx[label]
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# 图像变换
image_transforms = [transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomAffine(degrees=10),
        transforms.RandomRotation(degrees=(-180,180)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomApply(image_transforms, p=1.0),
        transforms.RandomChoice(image_transforms),
        transforms.Resize((256,256)),
        #transforms.RandomChoice(image_transforms),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 数据路径
data_dir = '../dataSets/dataMarrowCells/'

image_datasets = {x: LoadMyDataset(txtdata=os.path.join(data_dir, x + '.txt'),
                                   transform=data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""----------------------Write THE evalution index-----------------------"""
def write_to_excel(arr_1,arr_2,sava_path):

    writer = pd.ExcelWriter(sava_path)
    df1 = pd.DataFrame(arr_1)
    df2 = pd.DataFrame(arr_2)
    df1.to_excel(writer,sheet_name='train',header=False,index=False)
    df2.to_excel(writer,sheet_name='val',header=False,index=False)
    writer.close()

    return

"""------------------------------END-------------------------------------"""

"""----------------------Trainin the model-----------------------------"""


def train_model(model, criterion, optimizer, scheduler, modelName, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0
    train_index_arrays = [["epoch", "loss", "acc", "acc_class0","acc_class1","auc", "precision", "recall", "F1 score"]]
    eval_index_arrays = [["epoch", "loss", "acc","acc_class0","acc_class1","auc", "precision", "recall", "F1 score"]]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            y_true_list = []
            y_predict_list = []
            y_positive_score_list = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # softmax
                    softmax_layer = nn.Softmax(dim=1)
                    softmax_outputs = softmax_layer(outputs)
                    probability, preds = torch.max(softmax_outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                predict = preds.cpu().detach().numpy()
                probability = probability.cpu().detach().numpy()
                for i in range(len(predict)):
                    if predict[i] == 0:
                        probability[i] = 1 - probability[i]
                    y_true_list.append(int(labels[i]))
                    y_predict_list.append(int(predict[i]))
                    y_positive_score_list.append(float(probability[i]))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()  # 调整学习率

            epoch_loss = round(running_loss / dataset_sizes[phase],4)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            acc = round(float(epoch_acc.cpu().detach().numpy()),4)
            # 每类分类精度
            acc_for_each_class = metrics.precision_score(y_true_list, y_predict_list, average=None)
            acc_class0 = round(acc_for_each_class[0],4)
            acc_class1 = round(acc_for_each_class[1],4)
            # 计算auc的值
            fpr, tpr, thresholds = metrics.roc_curve(y_true_list, y_positive_score_list, pos_label=1)
            roc_auc = round(metrics.auc(fpr, tpr),4)
            # 查准率
            precision = metrics.precision_score(y_true_list, y_predict_list, average='macro')
            precision = round(precision,4)
            # 查全率
            recall = metrics.recall_score(y_true_list, y_predict_list, average='micro')
            recall = round(recall,4)
            # F1 score
            f1 = metrics.f1_score(y_true_list, y_predict_list, average='weighted')
            f1 = round(f1,4)

            print('{} Loss: {:.4f} Acc: {:.4f} roc_auc:{:.4f}'.format(phase, epoch_loss, epoch_acc, roc_auc))

            if phase == 'train':
                train_index_arrays.append([epoch,epoch_loss, acc, acc_class0, acc_class1, roc_auc, precision, recall, f1])

            if phase == 'val':
                eval_index_arrays.append([epoch, epoch_loss, acc, acc_class0, acc_class1, roc_auc, precision, recall, f1])
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and roc_auc > best_auc:
                best_auc = roc_auc
                best_auc_model = copy.deepcopy(model.state_dict())

            # 保存模型和日志
            writer = SummaryWriter(log_dir='logs')
            writer.add_scalar('data/{} loss'.format(phase), epoch_loss, epoch)
            writer.add_scalar('data/{} acc'.format(phase), epoch_acc, epoch)

        writer.export_scalars_to_json("./logs/all_scalars.json")
        writer.close()
    save_excel_dir = './indicators'
    if not os.path.exists(save_excel_dir):
        os.makedirs(save_excel_dir)
    sava_path = save_excel_dir + '/DensenNet121.xlsx'
    write_to_excel(train_index_arrays, eval_index_arrays, sava_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    # load best model weights
    model.load_state_dict(best_model_wts)
    acc_checkpoint_path = MODEL_PATH + modelName + '_best_acc_model.pth'
    torch.save(best_model_wts, acc_checkpoint_path)
    model.load_state_dict(best_auc_model)
    auc_checkpoint_path = MODEL_PATH + modelName + '_best_auc_model.pth'
    torch.save(best_auc_model, auc_checkpoint_path)

    return model

"""--------------------------------------End--------------------------------------------"""

"""------------------------------Finetuning the convnet---------------------------------"""
# Load a pretrained model and reset final fully connected layer.
def main():

    model_ft = models.densenet121(pretrained=True) # 加载预训练网络
    num_ftrs = model_ft.classifier.in_features

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss(weight=None, reduction='mean')

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           modelName = 'DensenNet121', num_epochs=35)


if __name__ == '__main__':

    main()
