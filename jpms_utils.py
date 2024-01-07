"""
Utility methods for analyzing and refactoring imglib2 and dependent projects for JPMS modularization.
"""

import logging, os
from pathlib import Path
from typing import List, Set, Dict


DEFAULT_PROJECTS = ["imglib2",
                    "imglib2-algorithm",
                    "imglib2-algorithm-fft",
                    "imglib2-algorithm-gpl",
                    "imglib2-cache",
                    "imglib2-ij",
                    "imglib2-mesh",
                    "imglib2-realtransform",
                    "imglib2-roi",
                    "imglib2-roi-io",
                    "imglib2-cache-python",
                    "imglib2-imglyb",
                    "imglib2-unsafe"]

log = logging.getLogger(__name__)


class GAV:
    """
    Maven coordinate, which can contain "*" to match any group, artifact name, and/or version.
    """
    def __init__(self, g = '*', a = '*', v = '*'):
        self.g = g
        self.a = a
        self.v = v

    def __str__(self):
        return f"{self.g}:{self.a}:{self.v}"

    def __eq__(self, other):
        return isinstance(other, GAV) and self.g == other.g and self.a == other.a and self.v == other.v

    def matches(self, gav) -> bool:
        return (self.g == gav.g or self.g == '*' or gav.g == '*') and (
                self.a == gav.a or self.a == '*' or gav.a == '*') and (
                self.v == gav.v or self.v == '*' or gav.v == '*')

    _default_m2_repo = Path("~/.m2/repository").expanduser()

    def pomPath(self, m2_repo: Path = _default_m2_repo ) -> Path:
        return m2_repo / self.g.replace('.', '/') / self.a / self.v / f"{self.a}-{self.v}.pom"


def loadGAVList(file_path: Path) -> List[GAV]:
    gavs = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                g, a, v = line.strip().split(':')
                gavs.append(GAV(g,a,v))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return gavs


def strip_trailing_slash(directory):
    """
    Remove optional trailing slash from directory path.
    """
    return directory.rstrip('/') if directory.endswith('/') else directory


def project_directory(imglib_root: Path, project: str) -> Path:
    """
    Get directory paths from root directory and project name.

    For each project, directory is either 'imglib_root/project/' or 'project/', whichever exists.
    """
    for fn in [imglib_root / project, Path(project)]:
        log.debug(f"looking for '{fn}'")
        if os.path.exists(fn) and os.path.isdir(fn):
            return fn
    log.warning(f"'{project}' not found.")


def project_directories(imglib_root: Path, projects: List[str]) -> List[Path]:
    """
    Get directory paths from root directory and project names.

    For each project, directory is either 'imglib_root/project/' or 'project/', whichever exists.
    """
    directories = []
    for project in projects:
        directory = project_directory(imglib_root, project)
        if not directory is None:
            directories.append(directory)

    return directories


def getPackageNames(src_root: Path) -> Set[str]:
    """
    Get all packages occurring in the given src_root.

    :param src_root: source directory, for example '/path/to/project/src/main/java'
    """
    # List .java files
    pkg_dirs = [root for root, dirs, files in os.walk(src_root)
                if any(file.endswith('.java') and not file.endswith('package-info.java') for file in files)]

    # Remove directory prefix and replace "/" with "."
    pkg_dirs = [dir.replace(str(src_root), '') for dir in pkg_dirs]
    pkgs = set(dir.replace('/', '.') for dir in pkg_dirs)

    return pkgs


def longestCommonPrefix(strings: Set[str]):
    """
    Return the longest common prefix shared by all strings.
    """
    strs = sorted(strings)
    first, last = strs[0], strs[-1]
    for i in range(len(first)):
        if last[i] != first[i]:
            return first[:i-1]
    return first


def getPackagePrefix(src_root: Path) -> str:
    """
    Get the longest common package prefix of all classes in the given src_root.

    :param src_root: source directory, for example '/path/to/project/src/main/java'
    """
    pkgs = getPackageNames(src_root)
    return longestCommonPrefix(pkgs)


def getJavaFiles(directory: Path) -> List[str]:
    """
    Get full path of all .java files in directory.

    The file 'package-info.java' is excluded, if it exists.
    """
    return [f"{root}/{file}" for root, dirs, files in os.walk(directory) for file in files if file.endswith('.java') and not file.endswith('package-info.java')]


def getClassNames(src_root: Path) -> List[str]:
    """
    Get fully qualified name of all classes in the given src_root.

    :param src_root: source directory, for example '/path/to/project/src/main/java'
    """
    filenames = getJavaFiles(src_root)
    prefix = str(src_root) + "/"

    # Remove src_root prefix and ".java" extension, replace "/" with "."
    clzs = [fn[:-5].replace(prefix, '').replace('/', '.') for fn in filenames]
    return clzs


def findOccurencesInFile(file_path: str, clzs: List[str]) -> List[str]:
    """
    Given a list of fully qualified class names, return the sub-list of those occurring in the given file

    :param file_path: file name
    :param clzs: list of fully qualified class names
    :return: sub-list of classes occurring in the given file
    """

    # List[str]
    # class names that are referenced by this file.
    occurring = []

    try:
        with open(file_path, 'r') as file:
            content = file.read()
            for clz in clzs:
                for c in [';', '.', ' ', '(', ')']:
                    if content.find(f"{clz}{c}") >= 0:
                        occurring.append(clz)
                        break

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    return occurring


def findOccurences(project_root: Path, clzs: List[str]) -> Dict[str, List[str]]:
    """
    Find occurrences of classes in a project.

    :param project_root: root directory of the project to check (all java files in "src/main/java/" and "src/test/java/"
                         are checked)
    :param clzs: list of fully qualified class names to check
    :return: Dictionary mapping occurring class name to list of files that it occurs in. (Only classes that occur in at
             least one file are keys.)
    """

    # Dict[str, List[str]]
    # maps class name to the project files that reference it.
    # only class names that are referenced by at least one file occur as keys.
    clz_to_files = dict()

    for src_root in ["src/main/java/", "src/test/java/"]:
        directory = project_root / src_root
        for fn in getJavaFiles(directory):
            occurring_clzs = findOccurencesInFile(fn, clzs)
            for oclz in occurring_clzs:
                files = clz_to_files.get(oclz, [])
                if not files:
                    clz_to_files[oclz] = files
                files.append(fn)

    return clz_to_files
