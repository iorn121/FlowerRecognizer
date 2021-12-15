import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import optimizer
import torchvision
from torchvision import datasets, models, transforms

from model import *
from data.data import output_dataset_path_list, MyDataset


def main():
    num_classes = 17
    batch_size = 32
    num_epoch = 20

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)])
    img_path = "data/images"

    train_list, valid_list = output_dataset_path_list(img_path, 17, 0.9)

    print('Setting up data...')
    train_data = MyDataset(train_list, transform)
    valid_data = MyDataset(valid_list, transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=len(valid_data), shuffle=True)

    print('Create model...')
    feature_extract = True
    model_ft = initialize_model(num_classes, feature_extract)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print('\t', name)

    print('Start training...')
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_model(model_ft, train_dataloader, valid_dataloader, criterion,
                optimizer_ft, device, num_epochs=num_epoch, num_vals=len(valid_data))


def train_model(model, train_dataloader, valid_dataloader, criterion, device, num_epochs=25, num_val=0):
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-'*10)

        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
                dataloaders = train_dataloader
            else:
                model.eval()
                dataloaders = valid_dataloader

            running_loss = 0.0
            running_corrects = 0

            for (i, (inputs, labels)) in enumerate(dataloaders):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    print('({}/{}) Epoch:{}/{} Loss:{:.5f} AveLoss:{:.5f}'.format(
                        i, int(len(dataloaders)), epoch, num_epochs, loss.item(), running_loss/(i+1)))
                else:
                    print('phase:eval Loss:{:.5f} Accuracy:{:.5f}%'.format(
                        loss.item(), running_corrects.item()/num_val*100))
            if (epoch) % 5 == 0:
                print('save model')
                torch.save(model.state_dict(),
                           'save_model/model_{}.pth'.format(epoch))
                epoch_loss = running_loss / len(dataloaders.dataset)
                epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print()


if __name__ == '__main__':
    main()
