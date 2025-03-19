#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2024-12-29 18:23:07             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import os
import sys
from urllib.parse import urlparse

import cv2
import yt_dlp


def get_available_devices(number_of_devices=10, max_index=1000, verbose=False):
    """
    Returns a list of available video capture devices.

    Args:
        number_of_devices (int): The maximum number of devices to search for.
        max_index (int): The maximum index to search for devices.
        verbose (bool): If True, prints the list of found devices.

    Returns:
        list: A list of available video capture device indices.
    """
    index, found_devices = 0, 0
    devices = []
    while (found_devices <= number_of_devices) and (index < max_index):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            devices.append(index)
            found_devices += 1
        cap.release()
        index += 1
    if verbose:
        print(devices)
    return devices


def parse_video_device(input_device, YT_URL=False):
    """
    Parses the video device based on the provided input.

    Args:
        input_device (str): The device to be parsed. Can be a URL, file path, or integer.
        YT_URL (bool): If True, converts the URL to a YouTube-compatible format.

    Returns:
        int: The parsed video device number.
    """
    # Get video device (0 means camera on computer, sometimes maybe 1)
    if input_device is None:
        input_device = str(get_available_devices(number_of_devices=1)[0])
        print('[INFO] Use first found device as input device')

    # Check if input is an URL
    result = urlparse(input_device)
    if result.scheme and result.netloc:
        if YT_URL:
            input_device = convert_YT_URL(input_device)
            print(f'[INFO] Input URL (YouTube): {input_device}')
        else:
            print(f'[INFO] Input URL: {input_device}')
    # Check if input is a file
    elif os.path.isfile(input_device):
        print(f'[INFO] Input file: {input_device}')
    # Check if input is a device
    else:
        try:
            input_device = int(input_device)
            print(f'[INFO] Input device: {input_device}')
        except ValueError:
            sys.exit(f'[ERROR] Cannot parse input {input_device} ...')
    return input_device


def convert_YT_URL(input_URL: str) -> str:
    """
    Convert a YouTube URL to its MP4 format using yt-dlp.

    Args:
    input_URL (str): The URL of the YouTube video.

    Returns:
    str: The converted URL in MP4 format.
    """

    # Configure yt-dlp
    ydl_opts = {
        'format': 'best[ext=mp4]',  # Get best MP4 format
        'quiet': True,
    }

    # Get video info and URL
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(input_URL, download=False)
        input_URL = info['url']

    return input_URL


def toggle_bool_option(bool_option: bool) -> bool:
    """
    Toggle the boolean value of `bool_option`.

    Args:
    bool_option (bool): The boolean value to be toggled.

    Returns:
    bool: The toggled boolean value.
    """

    if bool_option is True:
        bool_option = False
    else:
        bool_option = True
    return bool_option


def cycle_options(current_option, options):
    """
    Cycles through a list of options starting from the given current option.

    Parameters:
    - current_option: The current option to start cycling from.
    - options: A list of available options.

    Returns:
    - The next option in the list after the current option.
    """

    # Check number of options
    if len(options) < 2:
        print(f'[ERROR] Too few options ({len(options):d}) in {options}')
        return current_option

    # Check if current option is in options
    try:
        current_index = options.index(current_option)
    except ValueError:
        print(f'[ERROR] Cannot find {current_option} in options {options}')
        return current_index

    # Index
    if current_index == len(options) - 1:
        next_index = 0
    else:
        next_index = current_index + 1

    return options[next_index]
