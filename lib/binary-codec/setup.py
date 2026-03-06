"""Setup script for virtuals-binary-codec.

Builds the Cython binary key codec extension. No external C libraries required -
this is pure Cython compiled to C++.
"""

from pathlib import Path

from setuptools import Extension, setup


SRC_PATH = Path("src/virtuals_binary_codec")


def find_cython_files(package_path: Path) -> list[Path]:
    """Find all .pyx files in a package directory."""
    return sorted(package_path.rglob("*.pyx"))


def create_extensions() -> list[Extension]:
    """Create Extension objects for Cython files."""
    pyx_files = find_cython_files(SRC_PATH)

    if not pyx_files:
        print("No .pyx files found - skipping Cython compilation")
        return []

    try:
        from Cython.Build import cythonize
    except ImportError:
        print("WARNING: Cython not found, skipping extension build")
        print("  Install with: pip install Cython>=3.0")
        return []

    extensions = []
    for pyx_file in pyx_files:
        rel_path = pyx_file.relative_to("src")
        module_parts = [*list(rel_path.parts[:-1]), rel_path.stem]
        module_name = ".".join(module_parts)

        ext = Extension(
            name=module_name,
            sources=[str(pyx_file)],
            language="c++",
        )
        extensions.append(ext)

    cythonized = cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "embedsignature": True,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
            "overflowcheck": False,
        },
        annotate=False,
    )

    return cythonized


def main() -> None:
    """Main setup function."""
    extensions = create_extensions()
    setup(ext_modules=extensions)


if __name__ == "__main__":
    main()
