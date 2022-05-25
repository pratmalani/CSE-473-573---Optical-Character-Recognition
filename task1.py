"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(image)

    return image


def show_image(image, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', image)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    argmns = parser.parse_args()
    return argmns


import copy


def ocr(test_image, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.

    """
    # TODO Add your code here. Do not modify the return and input arguments

    original_image = copy.deepcopy(test_image)

    feature = enrollment(characters)

    bbox = detection(test_image)

    final = recognition(original_image, feature, bbox)

    return final
    # raise NotImplementedError


def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.

    feature_enrolment = []
    count = 0
    while count < len(characters):
        temp = characters[count]
        # Tried resizing with different sizes but got the best accuracy with this.
        temp[1] = cv2.resize(temp[1], (30, 30))

        feature_enrolment.append([temp[0], temp[1]])
        cv2.imwrite('./data/Resizedpixel' + temp[0] + '.png', temp[1])
        count += 1
        # show_image(temp[1], 0)
    print(len(feature_enrolment))

    return feature_enrolment
    # raise NotImplementedError


def detection(test_image):
    """
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    for index, val in np.ndenumerate(test_image):
        x, y = index[0], index[1]
        if test_image[x][y] <= 127:
            test_image[x][y] = 1
        else:
            test_image[x][y] = 0

    # r,c = test_image.shape

    currchar_label_mat = np.full((test_image.shape[0], test_image.shape[1]), 0)
    current_label = 0

    def label_marking(i, j):

        if (test_image[i][j] == 0):
            return
        if currchar_label_mat[i][j]:
            return

        currchar_label_mat[i][j] = current_label

        label_marking(i + 1, j + 0)
        label_marking(i + 0, j + 1)
        label_marking(i - 1, j + 0)
        label_marking(i + 0, j - 1)

    for index, val in np.ndenumerate(test_image):
        x, y = index[0], index[1]
        if currchar_label_mat[x][y] == 0:
            if test_image[x][y] == 1:
                current_label += 1
                label_marking(x, y)

    temp_res = []
    for i in range(0, current_label):
        temp_r = np.where(currchar_label_mat == i + 1)[0]
        temp_c = np.where(currchar_label_mat == i + 1)[1]
        temp_r = [int(x) for x in temp_r]
        temp_c = [int(x) for x in temp_c]

        temp_res.append({'bbox': [min(temp_c), min(temp_r), max(temp_r) - min(temp_r), max(temp_c) - min(temp_c)],
                         "name": "UNKNOWN"})

    return temp_res


def cropper(bbox, original_img):
    descriptors_of_test_img = []
    result = bbox
    cropped_img = []
    counter = 0
    while counter < len(bbox):
        dim1 = bbox[counter]['bbox'][1] - 1
        dim2 = bbox[counter]['bbox'][1] + bbox[counter]['bbox'][3] + 1
        dim3 = bbox[counter]['bbox'][0] - 1
        dim4 = bbox[counter]['bbox'][0] + bbox[counter]['bbox'][2] + 1

        cropped_img.append(cv2.resize(original_img[dim1:dim2, dim3:dim4], (30, 30)))

        cv2.rectangle(original_img, (bbox[counter]['bbox'][0], bbox[counter]['bbox'][1]), (
        bbox[counter]['bbox'][1] + bbox[counter]['bbox'][3], bbox[counter]['bbox'][0] + bbox[counter]['bbox'][2] + 1),
                      (255, 0, 0), 2)
        counter += 1

    return cropped_img, result


def correlation_coefficient(arr1, arr2):
    p = np.mean((arr1 - arr1.mean()) * (arr2 - arr2.mean()))
    s = arr1.std() * arr2.std()
    if s == 0:
        return 0
    else:
        p = p / s
        return p


def recognition(og_img, feature_enrolment, bbox):
    """
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    cropped, result = cropper(bbox, og_img)
    # print(cropped)
    cv2.imwrite('./data/exp.jpg', og_img)
    list_final = []
    for x in cropped:
        dict_temp = {}
        for y in feature_enrolment:
            # dict_temp[y[0]] =
            # print(y[1])
            # print(y[1].shape)
            # p = np.sum(y[1]*x)
            # m = np.sqrt(np.sum(np.square(y[1] * 2))) * np.sqrt(np.sum(np.square(x * 2)))
            # x= (x-np.mean(x)) / (np.std(x)*len(x))
            # y[1] = (x-np.mean(y[1])) / (np.std(y[1]))
            # print(np.correlate(x,y[1],'full'))
            coeff = correlation_coefficient(x, y[1])
            if coeff >= 0:
                dict_temp[y[0]] = coeff
            else:
                dict_temp[y[0]] = coeff
        # print(max(list_temp))
        list_final.append(dict_temp)
    for x in range(len(bbox)):
        d = list_final[x]
        v = list(d.values())
        k = list(d.keys())
        if max(v) > 0.47:
            bbox[x]['name'] = k[v.index(max(v))]
    # print(bbox)
    result = bbox

    ''' For loop to test all Cropped images from test_img and the extracted 
    features as above with the descriptor of each character that was enrolled.
    Later save the max enrolled character match less than threshold (320 selected)
    based on trial and error and save the label information in 'result'(name)'''
    # Tried implementing NCC.
    # show_image(og_img, 0)
    return result
    print(result)
    # raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    print(coordinates)
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")

    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_image = read_image(args.test_img)

    results = ocr(test_image, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
