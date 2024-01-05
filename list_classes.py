"""
List classes in project's src/main/java (fully qualified class names).
"""

import argparse
import logging
import os
from pathlib import Path

from jpms_utils import log, project_directory, getClassNames


# -- Main --

def main(imglib_root: Path, project: str, output_file: Path):

    log.info(f"imglib_root = '{imglib_root}'")
    log.info(f"projects = '{project}'")
    log.info(f"output_file = '{output_file}'")

    project_root = project_directory(imglib_root, project)
    clzs = getClassNames(project_root / "src/main/java")
    if output_file is None:
        for clz in clzs:
            print(clz)
    else:
        with open(output_file, 'w') as o:
            for clz in clzs:
                o.write(clz + '\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="List classes in project's src/main/java.")

    parser.add_argument("--imglib_root", default=None, help="path of directory containing imglib2 projects")
    parser.add_argument("--output", default=None, help="output file")
    parser.add_argument("project", help="project name or path")

    args = parser.parse_args()
    imglib_root = Path(args.imglib_root or os.getcwd()).expanduser()
    output_file = Path(args.output).expanduser() if args.output is not None else None
    projects = args.project

    main(imglib_root, projects, output_file)

