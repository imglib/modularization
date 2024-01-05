"""
Print automatic module name and longest package prefix for imglib2 projects.
"""

import logging, os, argparse

from typing import List
from pathlib import Path
from maven import POM
from jpms_utils import DEFAULT_PROJECTS, log, project_directory, getPackagePrefix


# -- Main --

def main(imglib_root: Path, projects: List[str]):

    log.info(f"imglib_root = '{imglib_root}'")
    log.info(f"projects = '{projects}'")

    if projects is None or len(projects) == 0:
        log.info("processing default project list")
        projects = DEFAULT_PROJECTS

    print("| project | automatic module name | package prefix |")
    print("|:--------|:----------------------|:---------------|")
    for project in projects:
        directory = project_directory(imglib_root, project)
        pom = POM(directory / "pom.xml")
        automatic_module_name = pom.properties.get('package-name', "UNDEFINED")
        package_prefix = getPackagePrefix(directory / "src/main/java")
        print(f"| {project} | `{automatic_module_name}` | `{package_prefix}` |")


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="List packages.")

    parser.add_argument("--imglib_root", default=None, help="path of directory containing imglib2 projects")
    parser.add_argument("projects", nargs="*", help="project names or paths")

    args = parser.parse_args()
    imglib_root = Path(args.imglib_root or os.getcwd()).expanduser()
    projects = args.projects

    main(imglib_root, projects)

