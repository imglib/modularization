"""
Find all usages of the specified classes.
"""

import argparse
import logging
from pathlib import Path
from typing import List

from jpms_utils import log, findOccurences


def loadClassList(file_path: Path) -> List[str]:
    clzs = []
    try:
        with open(file_path, 'r') as file:
            clzs = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return clzs


# -- Main --

def main(class_list: Path, project_root: Path):

    log.info(f"class_list = '{class_list}'")
    log.info(f"project_root = '{project_root}'")

    clzs = loadClassList(class_list)

    clz_to_files = findOccurences(project_root, clzs)
    if clz_to_files:
        for key, files in clz_to_files.items():
            files = [s.removeprefix(str(project_root)) for s in files]
            print(f"{key}:")
            for f in files:
                print(f"    {f}")


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Find all class usages.")

    parser.add_argument("class_list", type=str, help="file with fully qualified classnames")
    parser.add_argument("project", type=str, help="path of project to check")

    args = parser.parse_args()
    class_list = Path(args.class_list).expanduser()
    project_root = Path(args.project).expanduser()

    main(class_list, project_root)
