import subprocess
from pathlib import Path

def convert_h264_to_mp4(
    input_path,
    output_path=None,
    fps=30
):
    input_path = Path(r"C:\Coding\ML_DL\Datasets\1_04_A_042021014700AM.h264.h")

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    if output_path is None:
        output_path = input_path.with_suffix(".mp4")

    command = [
        "ffmpeg",
        "-y",                    # overwrite output
        "-framerate", str(fps),  # REQUIRED for raw .h264
        "-i", str(input_path),
        "-c:v", "libx264",       # universal codec
        "-pix_fmt", "yuv420p",   # max compatibility
        "-movflags", "+faststart",
        str(output_path)
    ]

    subprocess.run(command, check=True)
    print(f"Converted → {output_path}")

if __name__ == "__main__":
    convert_h264_to_mp4("input.h264", fps=30)
