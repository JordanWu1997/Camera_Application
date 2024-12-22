#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2024-12-18 22:16:05             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

from camera_application import put_text_to_canvas


def do_image_OCR(input_image, live_display=True, output=None):
    """  """

    # Check input
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
    elif isinstance(input_image, np.array):
        pass
    else:
        print('[ERROR] Unknown type of {input_image}')
        return

    # OCR
    result = pytesseract.image_to_boxes(image, lang='eng', config='--oem 1')

    # Post-processing
    chars_with_boxes = dict()
    for i, line in enumerate(result.split('\n')):
        cols = line.split(' ')
        if len(cols) == 1:
            continue
        char = cols[0]
        x1, y1 = int(cols[1]), int(cols[2])
        x2, y2 = int(cols[3]), int(cols[4])
        chars_with_boxes[char] = [x1, y1, x2, y2]

    # Visualization
    for char, boxes in chars_with_boxes.items():
        x1, y1, x2, y2 = boxes
        print(char, x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        put_text_to_canvas(image, char, top_left=(x1, y1), thickness=1)

    if live_display:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(canvas)
        plt.show()

    if output is not None:
        cv2.imwrite(output, image)


def main():

    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_images',
                        type=str,
                        nargs='+',
                        help='Input images to do OCR')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default=None,
                        help='Result output directory')
    args = parser.parse_args()

    # Init output dir
    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

    for image_path in args.input_images:

        # Check if input is a valid file
        if not os.path.isfile(image_path):
            print(f'[ERROR] {image_path} is not a file ...')
            continue

        # Init Output
        output = None
        if args.output_dir is not None:
            image_path_without_ext, image_ext = os.path.splitext(image_path)
            image_name_without_ext = os.path.basename(image_path_without_ext)
            output = f'{args.output_dir}/{image_name_without_ext}_OCR_out{image_ext}'

        # Main
        do_image_OCR(image_path, output=output)


if __name__ == '__main__':
    main()
