"""
Generate a target image from a txt Gaussian spec file.

Usage:
    python generate_target.py                                           # default example
    python generate_target.py data/txt/t3_sparse_colorful.txt       # specify txt file
    python generate_target.py data/txt/t3_sparse_colorful.txt --size 128 -o my_target.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from target_generators import render_txt_gaussians
from utils import resolve_device, save_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a txt Gaussian spec to an image")
    parser.add_argument(
        "txt_path",
        nargs="?",
        default="data/txt/t2_colorful_stars.txt",
        help="Path to the txt Gaussian spec file",
    )
    parser.add_argument("--size", type=int, default=256, help="Output image size (default: 256)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    txt_path = project_root / args.txt_path
    device = resolve_device("auto")

    image = render_txt_gaussians(txt_path=txt_path, image_size=args.size, device=device)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "outputs" / f"{txt_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output_path)

    print(f"Input:  {txt_path}")
    print(f"Size:   {args.size}x{args.size}")
    print(f"Saved:  {output_path}")


if __name__ == "__main__":
    main()
