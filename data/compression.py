"""H.264 compression helpers for frame datasets.

The project uses C0 for original frames and H.264 CRF 23 / 40 for C23 / C40.
FFmpeg must be installed and available on PATH for video encoding/decoding.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

COMPRESSION_CRFS = {"C0": None, "C23": 23, "C40": 40}


def validate_compression(level: str) -> str:
    level = level.upper()
    if level not in COMPRESSION_CRFS:
        raise ValueError(f"Unknown compression level: {level}. Use C0, C23, or C40.")
    return level


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(cmd: list[str]) -> None:
    if not ffmpeg_available():
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install FFmpeg before generating H.264 data."
        )
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def compress_video(input_video: str | Path, output_video: str | Path, crf: int,
                   preset: str = "medium") -> Path:
    """Encode a video with libx264 at the requested CRF."""
    input_video, output_video = Path(input_video), Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-i", str(input_video),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an", str(output_video),
    ])
    return output_video


def compress_frames_to_video(frame_dir: str | Path, output_video: str | Path,
                             crf: int, fps: float = 25.0,
                             pattern: str = "*.png") -> Path:
    """Encode an ordered image sequence into an H.264 video."""
    frame_dir, output_video = Path(frame_dir), Path(output_video)
    frames = sorted(frame_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No {pattern} frames found in {frame_dir}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    # FFmpeg image2 pattern is supported when filenames contain a numeric sequence.
    # For arbitrary filenames, create a temporary concat file instead.
    concat_file = output_video.with_suffix(".concat.txt")
    lines = []
    for frame in frames:
        safe = str(frame.resolve()).replace("'", "'\\''")
        lines.append(f"file '{safe}'")
        lines.append(f"duration {1.0 / fps:.8f}")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        _run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-vf", "fps=" + str(fps), "-c:v", "libx264", "-preset", "medium",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(output_video),
        ])
    finally:
        concat_file.unlink(missing_ok=True)
    return output_video


def extract_video_frames(video_path: str | Path, output_dir: str | Path,
                         image_ext: str = "png") -> int:
    """Decode every frame from a video into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in output_dir.iterdir() if p.is_file()}
    _run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vsync", "0", str(output_dir / f"frame_%06d.{image_ext}"),
    ])
    after = {p.name for p in output_dir.iterdir() if p.is_file()}
    return len(after - before)


def build_compressed_image_dataset(image_dir: str | Path, output_root: str | Path,
                                   levels: tuple[str, ...] = ("C23", "C40"),
                                   fps: float = 25.0) -> dict[str, Path]:
    """Create compressed frame directories from an extracted image dataset.

    The input layout is ``real/*.png`` and ``fake/*.png``. Each class is encoded
    as one H.264 sequence per compression level and decoded back into frames.
    """
    image_dir, output_root = Path(image_dir), Path(output_root)
    result: dict[str, Path] = {}
    for level in levels:
        level = validate_compression(level)
        if level == "C0":
            result[level] = image_dir
            continue
        crf = COMPRESSION_CRFS[level]
        level_root = output_root / level
        for label in ("real", "fake"):
            src = image_dir / label
            if not src.exists():
                raise FileNotFoundError(f"Missing class directory: {src}")
            # Encode the complete extracted sequence for the class. This is a
            # practical dataset-level helper; for video-faithful experiments,
            # call compress_video per source video before frame extraction.
            video = level_root / f"{label}.mp4"
            frames_out = level_root / label
            if not frames_out.exists() or not any(frames_out.glob("*.png")):
                compress_frames_to_video(src, video, crf=crf, fps=fps)
                extract_video_frames(video, frames_out, image_ext="png")
        result[level] = level_root
    return result
