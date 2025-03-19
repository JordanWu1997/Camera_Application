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
from PIL import Image, ImageDraw, ImageFont


def do_image_OCR(input_image, live_display=True, output=None):
    """
    Perform OCR on an input image and return the processed result.

    Parameters:
        input_image (str | np.ndarray): The path to the input image or a numpy array representing the image.
        live_display (bool, optional): Whether to display the processed image in real-time. Defaults to True.
        output (str | None, optional): The file path to save the processed image. If None, the image will not be saved.

    Returns:
        tuple[np.ndarray, list[str], list[list[int]]]: A tuple containing the processed image, a list of characters detected, and a list of bounding boxes for each character.
    """

    # Check input
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
    elif isinstance(input_image, np.array):
        pass
    else:
        print('[ERROR] Unknown type of {input_image}')
        return

    # OCR
    result_boxes = pytesseract.image_to_boxes(image,
                                              lang='eng',
                                              config='--oem 1 --psm 3')

    # Post-processing
    height, width, _ = image.shape
    chars, boxes, text = [], [], ''
    for i, line in enumerate(result_boxes.split('\n')):
        cols = line.split(' ')
        if len(cols) == 1:
            continue
        char = cols[0]
        x1, y1 = int(cols[1]), height - int(cols[4])
        x2, y2 = int(cols[3]), height - int(cols[2])
        chars.append(char)
        boxes.append([x1, y1, x2, y2])
    return image, chars, boxes


def visualize_OCR(
        image,
        chars,
        boxes,
        show_fig=True,
        output=None,
        font_path='./fonts/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf',
        font_size=20):
    """
    Visualize the results of OCR on an image.

    Args:
        image (np.ndarray or str): The input image. Can be a numpy array or a string representing an image file path.
        chars (list): A list of characters detected by OCR.
        boxes (list): A list of bounding boxes for each character, where each box is a tuple of four integers (x1, y1, x2, y2).
        show_fig (bool): Whether to display the result in a figure. Default is True.
        output (str): The path to save the resulting image. If None, the image will not be saved.
        font_path (str): The path to the font file used for drawing characters. Default is './fonts/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf'.
        font_size (int): The size of the font used for drawing characters. Default is 20.

    Returns:
        None
    """
    # Load image
    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, str):
        image = cv2.imread(image)
    else:
        print(f'[ERROR] Not supported image type {type(image)}')
        return

    image = Image.fromarray(image.astype('uint8')).convert('RGB')

    # Try to load a font, use default if not provided or fails
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Default font
            font = ImageFont.load_default()
    except Exception:
        print("Error loading font, using default")
        font = ImageFont.load_default()

    # Create a blank canvas with the same size as the original image
    blank = Image.new('RGB', image.size, (255, 255, 255))
    char_draw = ImageDraw.Draw(blank)
    box_draw = ImageDraw.Draw(image)

    # Draw the bounding boxes and characters on both images
    for char, (x1, y1, x2, y2) in zip(chars, boxes):
        # Draw rectangle on original image
        box_draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=1)
        # Draw rectangle on blank image
        char_draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=1)
        # Add character in middle of the box on blank image
        char_draw.text(((x1 + x2) // 2, (y1 + y2) // 2),
                       char,
                       fill="black",
                       font=font,
                       anchor="mm")
    # Concatenate original image and result image
    canvas = np.concatenate((image, blank), axis=1)

    # Plot result canvas
    plt.imshow(canvas)
    if show_fig:
        plt.show()
    if output is not None:
        plt.savefig(output, image)


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
        image, chars, boxes = do_image_OCR(image_path, output=output)
        visualize_OCR(image, chars, boxes, show_fig=True)


if __name__ == '__main__':
    main()
