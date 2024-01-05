"""
List GAVs for all potential dependent projects from pom_scijava.

Artefact versions are those given in the specified pom-scijava. We only really care about having *some* valid version
because we want to look into an existing POM to find the GitHub repository for the dependent project. We will then clone
the project and look at the main/master branch.
"""

import argparse
import logging
import re
from pathlib import Path

from jpms_utils import log, GAV
from maven import POM


# -- Main --

def main(pom_scijava_path: Path, output_file: Path):

    log.info(f"pom_scijava_path = '{pom_scijava_path}'")
    log.info(f"output_file = '{output_file}'")

    pom_scijava = POM(pom_scijava_path)
    deps = pom_scijava.dependencies(True)

    def resolve_dependency_version(version: str) -> str:
        """
        Resolve the actual dependency version from the given string, which could be a version number or a version
        property ("${property}"). Version properties are resolved recursively.
        """
        pattern = r'\${(.*?)}'
        match = re.search(pattern, version)
        while match:
            property = match.group(1)
            version = pom_scijava.properties.get(property)
            if version:
                match = re.search(pattern, version)
            else:
                # log.warning(f"version property '{property}' not found")
                return f"version property '{property}' not found"
        return version

    includeGAVs = [
        GAV("ca.mcgill"),
        GAV("io.scif"),
        GAV("jitk"),
        GAV("mpicbg"),
        GAV("net.imagej"),
        GAV("net.imglib2"),
        GAV("net.preibisch"),
        GAV("org.bonej"),
        GAV("org.janelia.saalfeldlab"),
        GAV("org.janelia"),
        GAV("org.morphonets"),
        GAV("org.scijava"),
        GAV("sc.fiji")]

    excludeGAVs = [
        GAV("net.imagej", "ij"),
        GAV("org.scijava", "j3dcore"),
        GAV("org.scijava", "j3dutils"),
        GAV("org.scijava", "jep"),
        GAV("org.scijava", "junit-benchmarks"),
        GAV("org.scijava", "vecmath")]

    def isIncluded(gav: GAV) -> bool:
        return any(gav.matches(g) for g in includeGAVs) and not any(gav.matches(g) for g in excludeGAVs)

    gavs = []
    for dependency in deps:
        g = dependency.groupId
        a = dependency.artifactId
        v = resolve_dependency_version(dependency.version)
        gav = GAV(g, a, v)
        if isIncluded(gav):
            gavs.append(gav)

    if output_file is None:
        for gav in gavs:
            print(gav)
    else:
        with open(output_file, 'w') as o:
            for gav in gavs:
                o.write(str(gav) + '\n')


if __name__ == '__main__':

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Extract GAVs of potentially dependent projects from pom_scijava.")

    parser.add_argument("--output", default=None, help="output file")
    parser.add_argument("pom_scijava_path", help="path of pom_scijava pom.xml")

    args = parser.parse_args()
    pom_scijava_path = Path(args.pom_scijava_path).expanduser()
    output_file = Path(args.output).expanduser() if args.output is not None else None

    main(pom_scijava_path, output_file)

