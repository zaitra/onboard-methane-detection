"""python build_package_for_deployment.py --parts inference processing onground --output satellite_pkg"""
import argparse
import shutil
from pathlib import Path
import textwrap

BASE_FILES = ["__init__.py", "mag1c_sas_base.py"]
OPTIONAL_DIRS = ["inference", "onground", "processing"]

def create_package(parts, output_dir, source_dir):
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()

    pkg_name = "onboard_methane_detection"

    if output.exists():
        shutil.rmtree(output)

    pkg_dir = output / pkg_name
    pkg_dir.mkdir(parents=True)

    # copy base files
    for f in BASE_FILES:
        src = source / f
        if not src.exists():
            raise FileNotFoundError(f"Missing base file: {src}")
        shutil.copy2(src, pkg_dir / f)

    # copy selected optional directories
    for part in parts:
        if part not in OPTIONAL_DIRS:
            raise ValueError(f"{part} is not a valid optional module")
        src = source / part
        if src.exists():
            shutil.copytree(src, pkg_dir / part)

    # create pyproject.toml
    pyproject = textwrap.dedent(f"""
    [project]
    name = "{pkg_name}"
    version = "0.1.0"
    description = "Minimal satellite deployment build"

    [tool.setuptools.packages.find]
    include = ["{pkg_name}*"]
    """)

    (output / "pyproject.toml").write_text(pyproject)

    # create minimal README (pip sometimes warns without it)
    (output / "README.md").write_text("# Satellite deployment build")

    print("Package created at:", output)
    print("To install:")
    print(f"  pip install {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts",
        nargs="*",
        default=[],
        help="Optional subpackages to include (inference, onground, processing)",
    )
    parser.add_argument(
        "--output",
        default="satellite_package",
        help="Output directory for generated package",
    )
    parser.add_argument(
        "--source",
        default="onboard_methane_detection",
        help="Source package directory",
    )

    args = parser.parse_args()

    create_package(args.parts, args.output, args.source)


if __name__ == "__main__":
    main()