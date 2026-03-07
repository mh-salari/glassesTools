"""Deploy default validation poster config, markers, and PDF."""

import importlib.resources
import pathlib
import shutil

from ... import aruco


def deploy_config(output_dir: str | pathlib.Path, overwrite: bool = False) -> list[str]:
    """Copy default validation config files to output_dir.

    Copies ``markerPositions.csv``, ``targetPositions.csv``, and
    ``validationSetup.txt`` from the bundled package, then calls
    :func:`deploy_maker` to also set up the poster TeX file and marker
    images.

    Args:
        output_dir: Directory to copy config files into.
        overwrite: If True, overwrite existing files.

    Returns:
        List of filenames that were NOT copied (already existed and
        ``overwrite`` is False).

    Raises:
        RuntimeError: If ``output_dir`` does not exist.

    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f'the requested directory "{output_dir}" does not exist')

    # copy over all config files
    not_copied: list[str] = []
    for r in ["markerPositions.csv", "targetPositions.csv", "validationSetup.txt"]:
        out_file = output_dir / r
        if out_file.exists() and not overwrite:
            not_copied.append(r)
            continue
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, out_file)

    not_copied.extend(deploy_maker(output_dir))
    return not_copied


def deploy_maker(output_dir: str | pathlib.Path, overwrite: bool = False) -> list[str]:
    """Copy the poster TeX file and generate marker images to output_dir.

    Args:
        output_dir: Directory to copy files into.
        overwrite: If True, overwrite existing files.

    Returns:
        List of filenames that were NOT copied (already existed and
        ``overwrite`` is False). Marker images are always regenerated.

    Raises:
        RuntimeError: If ``output_dir`` does not exist.

    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f'the requested directory "{output_dir}" does not exist')

    # copy over all files
    not_copied: list[str] = []
    for r in ["poster.tex"]:
        out_file = output_dir / r
        if out_file.exists() and not overwrite:
            not_copied.append(r)
            continue
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, out_file)

    deploy_marker_images(output_dir)  # N.B. these can be safely overwritten
    return not_copied


def deploy_marker_images(output_dir: str | pathlib.Path) -> None:
    """Generate and store ArUco marker images for the default validation poster.

    Creates an ``all-markers`` subdirectory inside ``output_dir`` and
    writes 1000 px marker images using the default validation setup's
    ArUco dictionary and border settings.

    Args:
        output_dir: Parent directory where the ``all-markers`` folder
            will be created.

    """
    from ..config import get_validation_setup

    output_dir = pathlib.Path(output_dir) / "all-markers"
    if not output_dir.is_dir():
        output_dir.mkdir()

    # get validation setup
    validation_setup = get_validation_setup()

    # generate and store the markers
    aruco.deploy_marker_images(
        output_dir, 1000, validation_setup["arucoDictionary"], validation_setup["markerBorderBits"]
    )


def deploy_default_pdf(output_file_or_dir: str | pathlib.Path) -> None:
    """Copy the default poster PDF to the specified file or directory.

    If ``output_file_or_dir`` is a directory, the PDF is written as
    ``poster.pdf`` inside it.

    Args:
        output_file_or_dir: Destination file path or directory.

    """
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir /= "poster.pdf"

    with importlib.resources.path(__package__, "poster.pdf") as p:
        shutil.copyfile(p, str(output_file_or_dir))
