from pathlib import Path
from typing import Dict

from jpms_utils import project_directory, getJavaFiles, getClassNames


def replaceClassNamesInFile(file_path: str, clz_map: Dict[str, str]) -> None:
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            modified_content = content
            for old_clz, new_clz in clz_map.items():
                for c in [';', '.', ' ', '(', ')']:
                    modified_content = modified_content.replace(f"{old_clz}{c}", f"{new_clz}{c}")

        if content != modified_content:
            print(f"writing modified {file_path}")
            with open(file_path, 'w') as file:
                file.write(modified_content)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")


def replaceClassNames(project_root: Path, clz_map: Dict[str, str]) -> None:
    for src_root in ["src/main/java/", "src/test/java/"]:
        directory = project_root / src_root
        for fn in getJavaFiles(directory):
            replaceClassNamesInFile(fn, clz_map)





prefix = "/Users/pietzsch/code/imglib"
project = "imglib2-ij"

directory = project_directory(Path(prefix), project)
clzs = getClassNames(directory / "src/main/java")
print(f"\n\n{project}")
for clz in clzs:
    print(f"  {clz}")
    clz.replace("net.imglib2.img", "net.imglib2.ij")
print()

# just for testing: replace package prefix
# in actual use, probably try matching class names of original and modified repository
clz_map : Dict[str, str] = {clz : clz.replace("net.imglib2.img", "net.imglib2.ij") for clz in clzs }
for old, new in clz_map.items():
    print(f"  {old} --> {new}")


project_root = Path("/Users/pietzsch/workspace/PreibischLab/multiview-reconstruction/")
replaceClassNames(project_root, clz_map)
