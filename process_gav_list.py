"""
Locate and clone GitHub repos for specified GAVs and check for occurrences of specified classes.
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from find_class_usage import loadClassList
from jpms_utils import log, findOccurences, loadGAVList
from maven import POM


# -- Main --

def main(gav_file: Path, class_list: Path, run_mvn_dependency_get: bool, force_repo_clone: bool):

    log.info(f"gav_file = '{gav_file}'")
    log.info(f"class_list = '{class_list}'")

    gavs = loadGAVList(gav_file)
    clzs = loadClassList(class_list)

    dependent_gavs = []

    for gav in gavs:

        log.info(f"processing {gav}")

        # download the POM from maven.scijava.org
        if run_mvn_dependency_get:
            subprocess.run(["mvn", "dependency:get", f"-Dartifact={gav}:pom", "-DremoteRepositories=scijava::::https://maven.scijava.org/content/groups/public", "-Dtransitive=false"], check=True)

        # extract the GitHub URL
        pom = POM(gav.pomPath())
        gh_repo = pom.scmURL.replace('https://github.com/', '')
        log.info(f"gh_repo = '{gh_repo}'")

        # clone into ./work/
        repo_path = Path(f"./work/{gh_repo}")
        log.info(f"repo_path = '{repo_path}'")
        if force_repo_clone and repo_path.exists():
            shutil.rmtree(repo_path)
        if not repo_path.exists():
            subprocess.run(["gh", "repo", "clone", gh_repo, repo_path, "--", "--depth", "1"], check=True)

        # find imglib2-ij classes
        clz_to_files = findOccurences(repo_path, clzs)
        if clz_to_files:
            dependent_gavs.append(gav)

    for gav in dependent_gavs:
        print(gav)


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Process GAV list.")

    parser.add_argument("gav_file", help="file with list of GAVs")
    parser.add_argument("class_list", type=str, help="file with fully qualified classnames")
    parser.add_argument("--no_mvn_dependency_get", dest='run_mvn_dependency_get', action='store_false', help="whether to run mvn dependency:get for each GAV")
    parser.add_argument("--no_force_repo_clone", dest='force_repo_clone', action='store_false', help="whether to re-clone repos that are already present in work directory")


    args = parser.parse_args()
    run_mvn_dependency_get = args.run_mvn_dependency_get
    force_repo_clone = args.force_repo_clone
    gav_file = Path(args.gav_file).expanduser()
    class_list = Path(args.class_list).expanduser()

    print(f"run_mvn_dependency_get = {run_mvn_dependency_get}")
    print(f"force_repo_clone = {force_repo_clone}")

    main(gav_file, class_list, run_mvn_dependency_get, force_repo_clone)

