__author__ = "Yarik Menchaca Resendiz"

import argparse
from xml.dom import minidom
from os.path import join
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_text_and_labes (string: str, remove_cause=True) -> tuple:
    string = string.replace("<\\", "</")
    # removing <cause> tags
    if remove_cause:
        string = string.replace('<cause>', '').replace('</cause>', '')
    tree = minidom.parseString(string)
    tag = tree.childNodes[0].tagName
    text = tree.childNodes[0].firstChild.nodeValue
    return text, tag


def create_dir_tuple(list_tuples: list, output_dir: str):
    for index, (text, emotion) in enumerate(list_tuples):
        path = Path("{}/{}".format(output_dir, emotion))
        path.mkdir(parents=True, exist_ok=True)
        with open(path.joinpath('{}.txt'.format(index)), 'w') as file:
            file.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', dest='output', type=str,
                        help='output directory', default='emotions')
    parser.add_argument('-i', '--input_path', dest='input', required=True,
                        type=str, help='input directory')
    args = parser.parse_args()

    with open(join(args.input, 'Emotion Cause.txt')) as file:
        dataset1 = [get_text_and_labes(line) for line in file.readlines()]

    dataset2 = list()
    with open(join(args.input, 'No Cause.txt')) as file:
        for line in file.readlines():
            try:
                dataset2.append(get_text_and_labes(line, False))
            except:
                pass

    dataset = dataset1 + dataset2

    train, test = train_test_split(dataset, test_size=0.2, random_state=32)

    create_dir_tuple(train, join(args.output, 'train'))
    create_dir_tuple(test, join(args.output, 'test'))
