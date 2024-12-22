#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2024-12-22 16:11:23             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import sys
import time

import cv2


def play_URL(input_URL):
    """  """

    # Capture URL
    cap = cv2.VideoCapture(input_URL)
    if not cap.isOpened():
        print(f'[ERROR] Cannot play {input_URL} ...')
        return
    else:
        print(f'[INFO] Start to play {input_URL} ...')

    # Main
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print('[ERROR] Cannot get image from source ... Retrying ...')
            time.sleep(0.1)

        # Refresh every 1 milisecond and detect pressed key
        key = cv2.waitKey(1)

        # Break loop when q or Esc is pressed
        if key == ord('q') or key == 27:
            break

        canvas = frame

        # Display
        cv2.imshow(f'Playing: {input_URL}', canvas)

    cap.release()
    cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_URL', type=str, help='Input URL to play')
    args = parser.parse_args()

    play_URL(args.input_URL)


if __name__ == '__main__':
    main()
