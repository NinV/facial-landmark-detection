"""
Download 300W dataset from https://github.com/jiankangdeng/MenpoBenchmark
Folder structure after unzip
-- 300w
    |-- Test        --> not use
    |-- Train
        |-- 300W_train.txt
        |-- image
            |-- afw             --> 337 images
            |-- helen
                |-- trainset    --> 2000 images
                |-- testset     --> 330 images
            |-- lfpw
                |-- trainset    --> 811 images
                |-- testset     --> 224 images
    |-- Validation
        |-- 300W_validation.txt
        |-- image           --> 135 images from IBUG

This code generate a train-test split based on:
    train and test split scheme from: https://arxiv.org/pdf/1803.04108.pdf
    This paper also using same train-test split: https://arxiv.org/pdf/2004.08190.pdf

Training: full set of AFW, training set of HELEN and LFPW. Total: 337 + 2000 + 811 = 3148 (images)
Testing:
    common subset: testing set of HELEN and LFPW. Total: 330 + 224 = 554 (images)
    challenge subset: IBUG (135 images)
    full set: common + challenge = 554 + 135 = 689 (images)

-- 300W_train_test_split
    |-- train
        | train_annotation.txt
        |-- images
    |-- test
        | test_common_annotation.txt
        | test_challenge_annotation.txt
        | test_full_annotation.txt
        |-- images
"""
import pathlib
import sys
import pandas as pd
from tqdm import tqdm


def is_common(row):
    print(row.image_name)
    return "testset" in row.image_name


def main():
    data_folder = pathlib.Path(sys.argv[1])

    root = data_folder.parent / "300W_train_test_split"
    train_images_folder = root / "train" / "images"
    test_images_folder = root / "test" / "images"
    root.mkdir(exist_ok=True)
    train_images_folder.mkdir(exist_ok=True, parents=True)
    test_images_folder.mkdir(exist_ok=True, parents=True)

    # training set
    csv_headers = ["image_name"]
    num_landmarks = 68
    for i in range(num_landmarks):
        csv_headers.extend(("x{}".format(i), "y{}".format(i)))

    df = pd.read_csv(data_folder / "Train/300W_train.txt", header=None, sep=" ")
    # print(df.head(5))
    train_df = pd.DataFrame(columns=csv_headers)
    common_df = pd.DataFrame(columns=csv_headers)

    num_common = 0
    num_train = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if is_common(row):
            num_common += 1
            # row.image_name =
            # common_df = common_df.append(row)
        else:
            # train_df = train_df.append(row)
            num_train += 1

    print(num_train, num_common)


if __name__ == '__main__':
    main()


