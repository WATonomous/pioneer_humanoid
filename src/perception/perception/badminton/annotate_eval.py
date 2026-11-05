"""Interactively annotate an eval manifest with shuttle centers and tags.

Controls:
  left click: mark the shuttle center and make the frame positive
  n: mark no shuttle / negative
  b: toggle blurred tag
  o: toggle occluded tag
  right arrow or space: save and advance
  left arrow: save and go back
  s: save
  q or escape: save and quit
"""

import argparse
import json
from pathlib import Path

import cv2


WINDOW_NAME = "shuttlecock eval annotation"


class Annotator:
    def __init__(self, rows: list[dict], manifest_path: Path):
        self.rows = rows
        self.manifest_path = manifest_path
        self.index = 0
        self.cursor = None

    @property
    def row(self) -> dict:
        return self.rows[self.index]

    def save(self) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as manifest:
            for row in self.rows:
                manifest.write(json.dumps(row) + "\n")

    def mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cursor = (x, y)
            self.row["label"] = "positive"
            self.row["center_x"] = float(x)
            self.row["center_y"] = float(y)

    def draw(self, image):
        display = image.copy()
        row = self.row
        label = row.get("label") or "UNLABELED"
        tags = row.setdefault("tags", [])
        text = f"{self.index + 1}/{len(self.rows)}  {label}  tags={','.join(tags) or '-'}"
        cv2.putText(display, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if row.get("center_x") is not None and row.get("center_y") is not None:
            center = (round(row["center_x"]), round(row["center_y"]))
            cv2.drawMarker(display, center, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        return display

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse)

        while self.rows:
            image = cv2.imread(self.row["image"])
            if image is None:
                raise RuntimeError(f"Could not read {self.row['image']}")

            cv2.imshow(WINDOW_NAME, self.draw(image))
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                self.save()
                break
            if key == ord("s"):
                self.save()
                continue
            if key == ord("n"):
                self.row["label"] = "negative"
                self.row["center_x"] = None
                self.row["center_y"] = None
                self.cursor = None
                continue
            if key in (ord("b"), ord("o")):
                tag = "blurred" if key == ord("b") else "occluded"
                tags = self.row.setdefault("tags", [])
                if tag in tags:
                    tags.remove(tag)
                else:
                    tags.append(tag)
                continue
            if key in (ord(" "), 83, 13):
                self.save()
                self.index = min(self.index + 1, len(self.rows) - 1)
                continue
            if key == 81:
                self.save()
                self.index = max(self.index - 1, 0)

        cv2.destroyAllWindows()


def load_rows(manifest_path: Path) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as manifest:
        rows = [json.loads(line) for line in manifest if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    Annotator(load_rows(manifest_path), manifest_path).run()
