from glob import glob
import os
import shutil

if __name__ == '__main__':
    label_name = list(open('label.txt'))
    os.mkdir('images')
    for i in range(len(label_name)):
        dir_name = label_name[i].replace('\n', '')
        os.mkdir(f'images/{dir_name}')
        for j in range(1, 81):
            shutil.move(
                f'jpg/image_{str(j+i*80).zfill(4)}.jpg', f'images/{dir_name}/{j}.jpg')
