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
import os
import sys
import time

import cv2

from image_OCR import get_OCR_in_chars, get_OCR_in_words, visualize_OCR_result
from utils import (get_available_devices, parse_video_device,
                   put_text_to_canvas, resize_image, toggle_bool_option)


def main():
    """  """

    # Input argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-devices',
                        action='store_true',
                        help='Get available device list')
    parser.add_argument('-i',
                        '--input_device',
                        default=None,
                        type=str,
                        help='Input device, file or strearming URL')
    parser.add_argument('-y',
                        '--YT_URL',
                        help='If input URL is youtube URL',
                        action='store_true')
    parser.add_argument('-r',
                        '--resize_ratio',
                        default=1.0,
                        type=float,
                        help='Ratio to resize live display')
    parser.add_argument('-f',
                        '--start_frame',
                        default=0,
                        type=int,
                        help='Frame to start')
    parser.add_argument('-s',
                        '--skip_frame',
                        type=int,
                        default=3,
                        help='Number of frame to skip')
    parser.add_argument('-l',
                        '--lang',
                        type=str,
                        default='chi_tra+eng',
                        help='Language for OCR')
    parser.add_argument('-c',
                        '--OCR_in_char',
                        action='store_true',
                        help='OCR in character-level')
    parser.add_argument('-fs',
                        '--font_size',
                        type=int,
                        default=30,
                        help='Fontsize of OCR result')
    args = parser.parse_args()

    # List available devices
    if args.list_devices:
        devices = get_available_devices(verbose=True)
        sys.exit(f'[INFO] Found devices: {devices}')

    # Init
    show_OSD = True

    # Get input device
    input_device = parse_video_device(args.input_device, YT_URL=args.YT_URL)

    # Capture URL
    cap = cv2.VideoCapture(input_device)
    if not cap.isOpened():
        print(f'[ERROR] Cannot play {input_device} ...')
        return
    else:
        print(f'[INFO] Start to play {input_device} ...')

    # Get frame property:  FPS
    input_FPS = cap.get(cv2.CAP_PROP_FPS)

    # Jump to frame to start
    if args.start_frame > 0:
        print(f'[INFO] Jump to frame {args.start_frame} ...')
        for _ in range(args.start_frame):
            _, _ = cap.read()

    # Main
    counter, skip_frame, playspeed = 0, args.skip_frame, 1
    font_size = args.font_size
    while True:

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print('[ERROR] Cannot get image from source ... Retrying ...')
            time.sleep(0.1)
        start = time.time()
        counter += 1

        # Speedup playing
        if playspeed > 1:
            for _ in range(int(playspeed) - 1):
                ret, frame = cap.read()
                if not ret:
                    print(
                        '[ERROR] Cannot get image from source ... Retrying ...'
                    )
                    time.sleep(0.1)
            start = time.time()
        elif playspeed < 1 and playspeed > 0:
            FPS = input_FPS * playspeed
            time.sleep(1 / FPS)
            ret, frame = cap.read()
            if not ret:
                print('[ERROR] Cannot get image from source ... Retrying ...')
                time.sleep(0.05)
            start = time.time()

        # Get image geometry: size
        height, width, _ = frame.shape

        # Resize frame
        if round(args.resize_ratio, 3) != 1.0:
            frame = resize_image(frame,
                                 width=width,
                                 height=height,
                                 resize_ratio=args.resize_ratio)

        # Refresh every 1 milisecond and detect pressed key
        key = cv2.waitKey(1)

        # Break loop when q or Esc is pressed
        if key == ord('q') or key == 27:
            break
        # Modify frame to skip
        if key == ord('='):
            skip_frame += 1
        if key == ord('-'):
            skip_frame -= 1
            if skip_frame < 1:
                print('[WARNING] Reached minimal skip_frame: 1')
                skip_frame = 1
        # Fast-forward: 1 sec
        if key == 83:  # Right
            fast_forward_sec = 1
            print(f'[INFO] Fast-forward {fast_forward_sec} secs ...')
            for _ in range(int(input_FPS * fast_forward_sec) - 1):
                _, _ = cap.read()
            continue
        # Fast-forward: 10 sec
        if key == 82:  # Up
            fast_forward_sec = 10
            print(f'[INFO] Fast-forward {fast_forward_sec} secs ...')
            for _ in range(int(input_FPS * fast_forward_sec) - 1):
                _, _ = cap.read()
            continue
        # Speedup playspeed
        if key == ord('s'):
            if playspeed >= 1.0:
                playspeed += 1.0
            else:
                playspeed += 0.25
        # Speeddown playspeed
        if key == ord('a'):
            if playspeed >= 2.0:
                playspeed -= 1.0
            else:
                playspeed -= 0.25
                playspeed = max(playspeed, 0.25)
        # Caption font size
        if key == ord('+'):
            font_size += 1
        if key == ord('_'):
            font_size -= 1
            if font_size < 1:
                print('[WARNING] Reached minimal OCR result font size: 1')
                font_size = 1

        # Perform object detection on an image
        if counter % skip_frame == 0 and counter > skip_frame:
            # OCR
            if args.OCR_in_char:
                result_dict = get_OCR_in_chars(frame, lang=args.lang)
            else:
                result_dict = get_OCR_in_words(frame, lang=args.lang)
            # Visualization
            canvas = visualize_OCR_result(frame,
                                          result_dict,
                                          output=None,
                                          show_fig=False,
                                          font_size=font_size)

            # Infer FPS
            infer_FPS = 1 / (time.time() - start)

        # Use previous result when object detection is ignored at current frame
        else:
            try:
                canvas = visualize_OCR_result(frame,
                                              result_dict,
                                              output=None,
                                              show_fig=False,
                                              font_size=font_size)
            except UnboundLocalError:
                canvas = frame

        # infer_FPS
        try:
            infer_FPS = infer_FPS
        except UnboundLocalError:
            infer_FPS = -1

        # Add OSD
        OSD_text = f'Input FPS: {input_FPS:.1f}, '
        OSD_text += f'Infer FPS: {infer_FPS:.1f}, '
        OSD_text += f'Playspeed: {playspeed:.2f}, '
        OSD_text += f'Infer every {skip_frame:d} frame, '
        OSD_text += f'FS: {font_size:d} '
        if key == 13:  # Enter
            show_OSD = toggle_bool_option(show_OSD)
        if show_OSD:
            put_text_to_canvas(canvas,
                               OSD_text,
                               top_left=(10, 30),
                               font_scale=0.5,
                               fg_color=(0, 255, 0),
                               thickness=1)

        # Display the annotated frame
        cv2.imshow(f"YOLOv11: {input_device}", canvas)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
