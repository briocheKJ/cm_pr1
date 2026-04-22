import argparse

from config import Config
from mode import set_mode
from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Gaussian Splatting image fitting")
    parser.add_argument(
        "--mode",
        choices=["student", "teacher"],
        default="student",
        help="student: uses student-implemented files; teacher: loads reference solutions",
    )
    args = parser.parse_args()
    set_mode(args.mode)
    train(Config())
