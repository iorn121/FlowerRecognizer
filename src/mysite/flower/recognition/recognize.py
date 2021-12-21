import torch
import numpy as np
from torchvision import transforms
import cv2

from flower.recognition.model import *


def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u


def main(img_path):
    model_path = 'flower/recognition/save_model/model_20.pth'
    label = ['Tulip',
             'Snowdrop',
             'LilyValley',
             'Bluebell',
             'Crocus',
             'Iris',
             'Tigerlily',
             'Daffodil',
             'Fritillary',
             'Sunflower',
             'Daisy',
             'ColtsFoot',
             'Dandelion',
             'Cowslip',
             'Buttercup',
             'Windflower',
             'Pansy']
    num_classes = 17
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    feature_extract = False
    model_ft = initialize_model(num_classes, use_pretrained=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_ft = model_ft.to(device)
    if torch.cuda.is_available():
        model_ft.load_state_dict(torch.load(model_path))
    else:
        model_ft.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))
    model_ft.eval()

    img_path = 'media/'+str(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = transform(img)
    img = img.view(1, 3, 224, 224)

    output = model_ft(img)
    output = output.detach().numpy()
    output = softmax(output)
    indices = np.argsort(-output)
    out_answer = []
    for i in range(5):
        out_answer.append(
            [label[indices[0, i]], round(100*output[0, indices[0, i]], 2)])
    return out_answer


if __name__ == '__main__':
    main()
