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

from acdc.engine.evaluation import Evaluation
from acdc.utils.common import load_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep learning trainer: evaluation script")

    parser.add_argument("conf", help="path to configuration file")
    parser.add_argument("result_folder", help="path to results folder")

    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        help="number of data loading workers (default: 0)",
    )
    args = parser.parse_args()
    args.cpu = True

    # load configuration file
    conf = load_conf(args.conf)

    evaluation = Evaluation(conf, args)
    evaluation.run()


if __name__ == "__main__":
    main()
