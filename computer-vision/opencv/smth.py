import re
from pathlib import Path

def c_header_to_h264_clean(header_path, output_path=None):
    header_path = Path(header_path)

    if output_path is None:
        output_path = header_path.with_name(
            header_path.name.replace(".h264.h", ".h264")
        )

    text = header_path.read_text(errors="ignore")

    # Extract hex bytes
    hex_bytes = re.findall(r'0x([0-9A-Fa-f]{2})', text)
    if not hex_bytes:
        raise ValueError("No hex bytes found")

    data = bytes(int(b, 16) for b in hex_bytes)

    out = bytearray()
    i = 0
    size = len(data)

    while i + 4 < size:
        # Look for length-prefixed NAL units
        nal_len = int.from_bytes(data[i:i+4], "big")
        if 0 < nal_len < 1_000_000 and i + 4 + nal_len <= size:
            out += b"\x00\x00\x00\x01"
            out += data[i+4:i+4+nal_len]
            i += 4 + nal_len
        else:
            i += 1  # resync

    if len(out) < 1000:
        raise ValueError("Reconstructed stream too small — input likely not valid H.264")

    output_path.write_bytes(out)
    print(f"Clean H.264 written to {output_path}")

if __name__ == "__main__":
    c_header_to_h264_clean(
        r"C:\Coding\ML_DL\Datasets\1_04_A_042021014700AM.h264.h"
    )
