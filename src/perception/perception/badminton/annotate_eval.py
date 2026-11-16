"""Interactively annotate an eval manifest with shuttle bounding boxes.

Controls:
    click-drag: draw the shuttle bounding box and make the frame positive
  n: mark no shuttle / negative
  right arrow or space: save and advance
  left arrow: save and go back
    + or =: zoom in around the current box
    - or _: zoom out
    0: reset zoom
  s: save
  q or escape: save and quit
"""

import argparse
import json
from pathlib import Path

import cv2


WINDOW_NAME = "shuttlecock eval annotation"


class Annotator:
    def __init__(self, rows: list[dict], manifest_path: Path, max_width: int, max_height: int):
        self.rows = rows
        self.manifest_path = manifest_path
        self.max_width = max_width
        self.max_height = max_height
        self.index = 0
        self.box_start = None
        self.box_end = None
        self.dragging = False
        self.mouse_point = None
        self.scale = 1.0
        self.banner_height = 72
        self.zoom = 1.0
        self.view_center = None
        self.crop_left = 0.0
        self.crop_top = 0.0

    @property
    def row(self) -> dict:
        return self.rows[self.index]

    def save(self) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as manifest:
            for row in self.rows:
                manifest.write(json.dumps(row) + "\n")

    def mouse(self, event, x, y, _flags, _param) -> None:
        if y < self.banner_height:
            self.mouse_point = None
            return
        original_point = (
            self.crop_left + x / self.scale,
            self.crop_top + (y - self.banner_height) / self.scale,
        )
        self.mouse_point = original_point
        if event == cv2.EVENT_LBUTTONDOWN:
            self.box_start = original_point
            self.box_end = original_point
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.box_end = original_point
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.box_end = original_point
            self.dragging = False
            x0, y0 = self.box_start
            x1, y1 = self.box_end
            left, right = sorted((x0, x1))
            top, bottom = sorted((y0, y1))
            if right - left < 1 or bottom - top < 1:
                self.row["label"] = "negative"
                self.row["bbox"] = None
                self.row["center_x"] = None
                self.row["center_y"] = None
                return
            self.row["label"] = "positive"
            self.row["bbox"] = [left, top, right, bottom]
            self.row["center_x"] = float((left + right) / 2)
            self.row["center_y"] = float((top + bottom) / 2)

    def update_view(self, width: int, height: int) -> None:
        if self.view_center is None:
            self.view_center = (width / 2, height / 2)
        crop_width = width / self.zoom
        crop_height = height / self.zoom
        center_x, center_y = self.view_center
        self.crop_left = max(0.0, min(width - crop_width, center_x - crop_width / 2))
        self.crop_top = max(0.0, min(height - crop_height, center_y - crop_height / 2))
        image_max_height = max(1, self.max_height - self.banner_height)
        self.scale = min(self.max_width / crop_width, image_max_height / crop_height)

    def set_zoom(self, zoom: float) -> None:
        if self.mouse_point is not None:
            self.view_center = self.mouse_point
        elif self.row.get("bbox"):
            left, top, right, bottom = self.row["bbox"]
            self.view_center = ((left + right) / 2, (top + bottom) / 2)
        self.zoom = max(1.0, min(4.0, zoom))

    def draw(self, image):
        height, width = image.shape[:2]
        self.update_view(width, height)
        crop_width = max(1, round(width / self.zoom))
        crop_height = max(1, round(height / self.zoom))
        crop = image[
            round(self.crop_top):round(self.crop_top) + crop_height,
            round(self.crop_left):round(self.crop_left) + crop_width,
        ]
        display = cv2.resize(crop, None, fx=self.scale, fy=self.scale)
        row = self.row
        label = row.get("label") or "UNLABELED"
        display = cv2.copyMakeBorder(
            display,
            self.banner_height,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        status = f"Frame {self.index + 1}/{len(self.rows)} | {label}"
        controls = "Drag box | +/- zoom | 0 reset | n none | arrows/space move | s save | q quit"
        cv2.putText(display, status, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.putText(display, controls, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        if row.get("center_x") is not None and row.get("center_y") is not None:
            center = (
                round((row["center_x"] - self.crop_left) * self.scale),
                round((row["center_y"] - self.crop_top) * self.scale) + self.banner_height,
            )
            cv2.drawMarker(display, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        if row.get("bbox"):
            left, top, right, bottom = row["bbox"]
            cv2.rectangle(
                display,
                (round((left - self.crop_left) * self.scale), round((top - self.crop_top) * self.scale) + self.banner_height),
                (round((right - self.crop_left) * self.scale), round((bottom - self.crop_top) * self.scale) + self.banner_height),
                (0, 255, 0),
                1,
            )
        if self.dragging and self.box_start and self.box_end:
            left, right = sorted((self.box_start[0], self.box_end[0]))
            top, bottom = sorted((self.box_start[1], self.box_end[1]))
            cv2.rectangle(
                display,
                (round((left - self.crop_left) * self.scale), round((top - self.crop_top) * self.scale) + self.banner_height),
                (round((right - self.crop_left) * self.scale), round((bottom - self.crop_top) * self.scale) + self.banner_height),
                (255, 255, 0),
                1,
            )
        if self.mouse_point is not None:
            mouse_x = round((self.mouse_point[0] - self.crop_left) * self.scale)
            mouse_y = round((self.mouse_point[1] - self.crop_top) * self.scale) + self.banner_height
            cross_size = 10
            cv2.line(display, (mouse_x - cross_size, mouse_y), (mouse_x + cross_size, mouse_y), (255, 255, 255), 1)
            cv2.line(display, (mouse_x, mouse_y - cross_size), (mouse_x, mouse_y + cross_size), (255, 255, 255), 1)
        return display

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.max_width, self.max_height)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse)

        while self.rows:
            image = cv2.imread(self.row["image"])
            if image is None:
                raise RuntimeError(f"Could not read {self.row['image']}")

            cv2.imshow(WINDOW_NAME, self.draw(image))
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if key in (ord("q"), 27):
                self.save()
                break
            if key == ord("s"):
                self.save()
                continue
            if key == ord("n"):
                self.row["label"] = "negative"
                self.row["bbox"] = None
                self.row["center_x"] = None
                self.row["center_y"] = None
                self.box_start = None
                self.box_end = None
                continue
            if key in (ord("+"), ord("=")):
                self.set_zoom(self.zoom * 1.5)
                continue
            if key in (ord("-"), ord("_")):
                self.set_zoom(self.zoom / 1.5)
                continue
            if key == ord("0"):
                self.set_zoom(1.0)
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
    parser.add_argument("--max_width", type=int, default=1280)
    parser.add_argument("--max_height", type=int, default=720)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    if args.max_width <= 0 or args.max_height <= 0:
        parser.error("--max_width and --max_height must be positive")
    Annotator(load_rows(manifest_path), manifest_path, args.max_width, args.max_height).run()
