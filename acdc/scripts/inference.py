__copyright__ = """
    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2022

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law.
"""

__license__ = "CC BY-NC-SA 3.0"

import argparse

from loguru import logger

from acdc.engine.inference import Inference
from acdc.utils.common import load_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep learning trainer: inference script")

    parser.add_argument("conf", help="path to configuration file")
    parser.add_argument("model", help="path to model checkpoint")

    parser.add_argument("-cpu", action="store_true", help="use cpu?")
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        help="number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training"
    )
    parser.add_argument(
        "-o", "--outputfolder", default="./output/", type=str, help="path for the ouput folder"
    )
    args = parser.parse_args()

    # load configuration file
    try:
        conf = load_conf(args.conf)

        inference = Inference(conf, args)
        inference.run()
    except:
        logger.exception("Exception")


if __name__ == "__main__":
    main()
