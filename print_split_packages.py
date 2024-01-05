"""
Print split packages and list of imglib2 projects they occur in.
"""

import logging, os, argparse

from typing import List, Dict, Set
from pathlib import Path
from jpms_utils import DEFAULT_PROJECTS, log, project_directory, getPackageNames


# -- Main --

def main(imglib_root: Path, projects: List[str]):

    log.info(f"imglib_root = '{imglib_root}'")
    log.info(f"projects = '{projects}'")

    if projects is None or len(projects) == 0:
        log.info("processing default project list")
        projects = DEFAULT_PROJECTS

    # maps project to set of packages
    pkg_index: Dict[str,Set[str],] = {}
    for project in projects:
        directory = project_directory(imglib_root, project)
        pkg_set = getPackageNames(directory)
        pkg_index[project] = pkg_set

    # maps package name to projects it occurs in
    pkg_to_projects: Dict[str,Set[str]] = {}
    for project in projects:
        for pkg in pkg_index[project]:
            prj = pkg_to_projects.get(pkg, set())
            if not prj:
                pkg_to_projects[pkg] = prj
            prj.add(project)

    # find split packages
    split_pkgs = [pkg for pkg, prj in pkg_to_projects.items() if len(prj) > 1]

    print("| package                    | projects                |")
    print("|:---------------------------|:------------------------|")
    for pkg in split_pkgs:
        print(f"| `{pkg}` | {', '.join(pkg_to_projects[pkg])} |")


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="List packages.")

    parser.add_argument("--imglib_root", default=None, help="path of directory containing imglib2 projects")
    parser.add_argument("projects", nargs="*", help="project names or paths")

    args = parser.parse_args()
    imglib_root = Path(args.imglib_root or os.getcwd()).expanduser()
    projects = args.projects

    main(imglib_root, projects)

