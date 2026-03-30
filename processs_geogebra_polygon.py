import re
import json
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"""\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"""
)

def parse_psline_file(text: str):
    """
    Parse lines like:
    \\psline[linewidth=2pt,linecolor=rvwvcq](-4.79,0.89)(-6.17,-0.15)

    Returns:
        vertices: list of [x, y]
        segments: list of [i, j]
    """
    raw_segments = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or "\\psline" not in line:
            continue

        m = LINE_RE.search(line)
        if not m:
            raise ValueError(f"Could not parse line {line_no}: {line}")

        x1, y1, x2, y2 = map(float, m.groups())
        raw_segments.append(((x1, y1), (x2, y2)))

    if not raw_segments:
        raise ValueError("No valid \\psline segments found.")

    vertices = []
    segments = []

    # Use exact parsed coordinates as keys; if you want tolerance-based matching,
    # we can change this later.
    vertex_to_index = {}

    def get_index(pt):
        if pt not in vertex_to_index:
            vertex_to_index[pt] = len(vertices)
            vertices.append([pt[0], pt[1]])
        return vertex_to_index[pt]

    for p1, p2 in raw_segments:
        i = get_index(p1)
        j = get_index(p2)
        segments.append([i, j])

    return {"vertices": vertices, "segments": segments}


def main():
    if len(sys.argv) != 3:
        print("Usage: python psline_to_json.py input.txt output.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    text = input_path.read_text(encoding="utf-8")
    data = parse_psline_file(text)

    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(data['vertices'])} vertices and {len(data['segments'])} segments to {output_path}")


if __name__ == "__main__":
    main()