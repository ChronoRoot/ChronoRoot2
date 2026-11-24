#!/usr/bin/env python3
"""CLI wrapper for the PyQt screening preview viewer."""

import argparse
import sys

from PyQt5.QtWidgets import QApplication

import chrono_root_backend  # noqa: F401
from analysis.utils.fileUtilities import getImages
import plant_viewer


def main():
    parser = argparse.ArgumentParser(description='Preview video sequence from images')
    parser.add_argument('--video-dir', required=True,
                        help='Directory containing the image sequence')
    parser.add_argument('--segmentation-dir',
                        help='Ignored; segmentation is loaded from video-dir/Segmentation like the main app')
    parser.add_argument('--time-delta', type=float, default=15.0,
                        help='Time in minutes between frames (default: 15)')
    args = parser.parse_args()

    try:
        conf = {'Images': args.video_dir, 'timeStep': args.time_delta}
        images, seg_files = getImages(conf)
        if not images:
            raise FileNotFoundError(
                f"No images found for {args.video_dir}. "
                "Check the folder or segmentation metadata input_path."
            )
        app = QApplication(sys.argv)
        window = plant_viewer.ChronoViewWindow(images, seg_files, None, conf)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error during preview: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
