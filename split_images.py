import os
import argparse
import random

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', '--labels_path',
                        help='The path to the labels csv file.', required=True)
    parser.add_argument('-imgs', '--images_path',
                        help='The path to the images to split into train_test'
                        + ' sets.', required=True)
    parser.add_argument('-test', '--test_percentage',
                        help='The percentage of testing images to create',
                        type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.labels_path)

    images = os.listdir(args.images_path)
    random.shuffle(images)
    index = len(images)*args.test_percentage // 100
    print(len(images), 'total images')
    print(len(images) - index, 'train images')
    print(index, 'test images')
    train = images[index:]
    test = images[:index]

    train_df = df[df.filename.isin(train)]
    test_df = df[df.filename.isin(test)]

    try:
        os.mkdir('{}train'.format(args.images_path))
    except FileExistsError:
        pass

    try:
        os.mkdir('{}test'.format(args.images_path))
    except FileExistsError:
        pass


    os.system('mv {} {}train/'.format(
        ' '.join([args.images_path + i for i in train]), args.images_path))
    os.system('mv {} {}test/'.format(
        ' '.join([args.images_path + i for i in test]), args.images_path))

    train_df.to_csv('train_labels.csv')
    test_df.to_csv('test_labels.csv')


if __name__ == "__main__":
    main()
