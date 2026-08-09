"""SmartPack: passive two-stage food-box inspection with one overhead camera.

The camera should be zoomed into the box opening. The entire frame is the
counting region; SmartPack does not ask volunteers to cross a virtual line or
place items in a smaller region of interest.

Workflow:
    1. Learn an empty-table reference at startup.
    2. Detect a box entering the view and count visible cans.
    3. Save the last settled can count when the box leaves.
    4. Load the chili/stew model and repeat for the same box.
    5. Save both station results to CSV and report PASS or CHECK COUNTS.

Default model order:
    can.pt -> chili-stew.pt

Controls:
    R       Recalibrate the empty work surface.
    Q / Esc Quit.

Requirements:
    python -m pip install ultralytics opencv-python

Example:
    python smartpack.py --camera 1 --device 0 --show-mask
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Station:
    name: str
    model_path: Path
    target: int


class StableCount:
    """Latch a detection count after it remains unchanged for several frames."""

    def __init__(self, required_frames: int):
        self.required_frames = required_frames
        self.value = 0
        self.candidate = 0
        self.candidate_frames = 0

    def reset(self) -> None:
        self.value = 0
        self.candidate = 0
        self.candidate_frames = 0

    def update(self, detected_count: int) -> int:
        if detected_count == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = detected_count
            self.candidate_frames = 1
        if self.candidate_frames >= self.required_frames:
            self.value = self.candidate
        return self.value


def parse_int_list(value: str, expected: int, label: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must contain comma-separated integers") from exc
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"{label} must contain exactly {expected} values")
    if any(number < 1 for number in values):
        raise argparse.ArgumentTypeError(f"every {label} value must be at least 1")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count cans and chili/stew in two sequential SmartPack stations."
    )
    parser.add_argument("--camera", type=int, default=0, help="USB/virtual camera index.")
    parser.add_argument(
        "--models",
        nargs=2,
        type=Path,
        default=[PROJECT_DIR / "can.pt", PROJECT_DIR / "chili-stew.pt"],
        metavar=("CAN", "CHILI_STEW"),
        help="Specialized YOLO weights in station order.",
    )
    parser.add_argument(
        "--targets",
        default="1,1",
        help="Required counts for cans,chili-stew (default: 1,1).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, for example 0 for the first GPU or cpu.",
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=45,
        help="Empty-table frames averaged at startup.",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=28,
        help="Minimum brightness difference from the empty-table reference.",
    )
    parser.add_argument(
        "--color-threshold",
        type=int,
        default=16,
        help="Minimum normalized-color difference; rejects brightness-only shadows.",
    )
    parser.add_argument(
        "--strong-pixel-threshold",
        type=int,
        default=45,
        help=(
            "Brightness difference that is accepted without a color change. "
            "This lets neutral cardboard differ from a similarly colored surface."
        ),
    )
    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=16,
        help="Minimum frame-to-frame brightness difference treated as motion.",
    )
    parser.add_argument(
        "--max-motion-ratio",
        type=float,
        default=0.012,
        help="Maximum moving-pixel fraction for the scene to count as settled.",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=12,
        help="Consecutive still frames required before accepting a new box.",
    )
    parser.add_argument(
        "--count-stable-frames",
        type=int,
        default=8,
        help="Settled frames with the same detection count before it is latched.",
    )
    parser.add_argument(
        "--presence-ratio",
        type=float,
        default=0.10,
        help="Changed fraction of the full frame required for box presence.",
    )
    parser.add_argument(
        "--absence-ratio",
        type=float,
        default=0.035,
        help="Changed fraction below which a box is considered absent.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path for an annotated MP4 recording.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=PROJECT_DIR / "smartpack_events.csv",
        help="CSV event log path.",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Display the OpenCV box-presence mask in a second window.",
    )
    args = parser.parse_args()
    args.targets = parse_int_list(args.targets, 2, "targets")
    return args


class BoxPresenceDetector:
    """Detect a box using full-frame change from an empty-table reference."""

    ENTRY_VOTES = 6
    EXIT_VOTES = 8

    def __init__(self, cv2_module, args: argparse.Namespace):
        self.cv2 = cv2_module
        self.args = args
        self.background_gray = None
        self.background_chroma = None
        self.previous_gray = None
        self.present = False
        self.enter_votes = 0
        self.exit_votes = 0
        self.stable_frames = 0

    def _gray(self, frame):
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        return self.cv2.GaussianBlur(gray, (7, 7), 0)

    def _chroma(self, frame):
        """Return per-pixel BGR proportions that are less sensitive to shadows."""

        blurred = self.cv2.GaussianBlur(frame, (7, 7), 0).astype("float32")
        channel_total = blurred.sum(axis=2, keepdims=True)
        chroma = blurred * (255.0 / (channel_total + 1.0))
        return chroma.clip(0, 255).astype("uint8")

    def reset_state(self) -> None:
        self.present = False
        self.enter_votes = 0
        self.exit_votes = 0
        self.stable_frames = 0

    def calibrate(self, camera) -> bool:
        print(f"Calibrating on {self.args.calibration_frames} empty-table frames...")
        gray_accumulator = None
        chroma_accumulator = None
        last_gray = None

        for index in range(self.args.calibration_frames):
            ok, frame = camera.read()
            if not ok:
                print("Camera frame failed during calibration.")
                return False

            gray = self._gray(frame)
            chroma = self._chroma(frame)
            if gray_accumulator is None:
                gray_accumulator = gray.astype("float32")
                chroma_accumulator = chroma.astype("float32")
            else:
                weight = 1.0 / (index + 1)
                self.cv2.accumulateWeighted(gray, gray_accumulator, weight)
                self.cv2.accumulateWeighted(chroma, chroma_accumulator, weight)
            last_gray = gray

            preview = frame.copy()
            self.cv2.rectangle(preview, (0, 0), (preview.shape[1], 58), (20, 20, 20), -1)
            self.cv2.putText(
                preview,
                f"KEEP VIEW EMPTY - calibrating {index + 1}/{self.args.calibration_frames}",
                (15, 37),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 220, 255),
                2,
                self.cv2.LINE_AA,
            )
            self.cv2.imshow("SmartPack", preview)
            if self.cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return False

        self.background_gray = self.cv2.convertScaleAbs(gray_accumulator)
        self.background_chroma = self.cv2.convertScaleAbs(chroma_accumulator)
        self.previous_gray = last_gray.copy()
        self.reset_state()
        print("Calibration complete. Move the box into the camera view.")
        return True

    def _presence_mask(self, gray, chroma):
        brightness_difference = self.cv2.absdiff(gray, self.background_gray)
        _, brightness_mask = self.cv2.threshold(
            brightness_difference,
            self.args.pixel_threshold,
            255,
            self.cv2.THRESH_BINARY,
        )

        color_difference = self.cv2.absdiff(chroma, self.background_chroma).max(axis=2)
        _, color_mask = self.cv2.threshold(
            color_difference,
            self.args.color_threshold,
            255,
            self.cv2.THRESH_BINARY,
        )

        # A normalized-color requirement by itself is excellent at rejecting
        # shadows, but it can erase tan/gray cardboard whose B:G:R proportions
        # resemble the empty surface. Accept either:
        #   1. a moderate brightness change supported by a color change, or
        #   2. a large brightness change, even when the object is neutral.
        color_supported_change = self.cv2.bitwise_and(brightness_mask, color_mask)
        _, strong_brightness_change = self.cv2.threshold(
            brightness_difference,
            self.args.strong_pixel_threshold,
            255,
            self.cv2.THRESH_BINARY,
        )
        mask = self.cv2.bitwise_or(color_supported_change, strong_brightness_change)

        # Five pixels removes camera noise without deleting thin box edges.
        kernel = self.cv2.getStructuringElement(self.cv2.MORPH_RECT, (5, 5))
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
        return self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel, iterations=2)

    def update(self, frame):
        height, width = frame.shape[:2]
        frame_area = max(1, width * height)
        gray = self._gray(frame)
        chroma = self._chroma(frame)
        mask = self._presence_mask(gray, chroma)
        presence_score = self.cv2.countNonZero(mask) / frame_area

        if self.previous_gray is None:
            motion_ratio = 1.0
        else:
            motion_difference = self.cv2.absdiff(gray, self.previous_gray)
            _, motion_mask = self.cv2.threshold(
                motion_difference,
                self.args.motion_threshold,
                255,
                self.cv2.THRESH_BINARY,
            )
            motion_kernel = self.cv2.getStructuringElement(self.cv2.MORPH_RECT, (3, 3))
            motion_mask = self.cv2.morphologyEx(
                motion_mask, self.cv2.MORPH_OPEN, motion_kernel
            )
            motion_ratio = self.cv2.countNonZero(motion_mask) / frame_area
        self.previous_gray = gray.copy()

        settled = motion_ratio <= self.args.max_motion_ratio
        event = None
        if not self.present:
            self.stable_frames = (
                min(self.args.settle_frames, self.stable_frames + 1) if settled else 0
            )
            if (
                self.stable_frames >= self.args.settle_frames
                and presence_score >= self.args.presence_ratio
            ):
                self.enter_votes += 1
            else:
                self.enter_votes = max(0, self.enter_votes - 1)

            if self.enter_votes >= self.ENTRY_VOTES:
                self.present = True
                self.enter_votes = 0
                self.exit_votes = 0
                event = "entered"
        else:
            if presence_score <= self.args.absence_ratio:
                self.exit_votes += 1
            else:
                self.exit_votes = max(0, self.exit_votes - 1)

            if self.exit_votes >= self.EXIT_VOTES:
                self.present = False
                self.enter_votes = 0
                self.exit_votes = 0
                self.stable_frames = 0
                event = "exited"

        return (
            self.present,
            event,
            mask,
            presence_score,
            motion_ratio,
            self.stable_frames,
        )


def append_event(
    log_path: Path,
    station: Station,
    action: str,
    count: int,
    detail: str = "",
) -> None:
    path = log_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["timestamp", "station", "action", "count", "detail"])
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                station.name,
                action,
                count,
                detail,
            ]
        )


def load_model(YOLO, station: Station):
    print(f"Loading {station.name} model: {station.model_path}")
    return YOLO(str(station.model_path))


def reset_tracker(model) -> None:
    """Force Ultralytics to build a fresh predictor and ByteTrack instance."""

    model.predictor = None


def count_color(count: int, target: int) -> tuple[int, int, int]:
    if count == target:
        return 70, 255, 120
    if count > target:
        return 40, 40, 255
    return 0, 220, 255


def draw_status(
    cv2,
    frame,
    stations: list[Station],
    saved_counts: list[int],
    active_index: int,
    box_present: bool,
    instantaneous_count: int,
    stable_count: int,
    presence_score: float,
    motion_ratio: float,
    stable_frames: int,
    settle_frames: int,
    workflow_complete: bool,
) -> None:
    panel_height = 132
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    current = stations[active_index]
    cv2.putText(
        frame,
        f"SmartPack | Station {active_index + 1}/{len(stations)}: "
        f"{current.name}  {stable_count}/{current.target}",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        count_color(stable_count, current.target),
        2,
        cv2.LINE_AA,
    )

    summary = " | ".join(
        f"{station.name}:{saved_counts[index]}/{station.target}"
        for index, station in enumerate(stations)
    )
    cv2.putText(
        frame,
        summary,
        (14, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    if workflow_complete:
        passed = all(
            count == station.target for count, station in zip(saved_counts, stations)
        )
        instruction = (
            "ALL TWO STATION COUNTS SAVED - PASS"
            if passed
            else "ALL TWO STATION COUNTS SAVED - CHECK COUNTS"
        )
    elif not box_present and stable_frames < settle_frames:
        instruction = f"Waiting for scene to settle ({stable_frames}/{settle_frames})"
    elif not box_present:
        instruction = "Move box into view and let everything settle"
    elif stable_count > current.target:
        instruction = "OVER TARGET - remove extra item and let the count settle"
    elif stable_count == current.target:
        instruction = "Target reached - remove box to save this station"
    else:
        instruction = (
            f"Visible now: {instantaneous_count} | "
            f"Saved when box leaves: {stable_count}"
        )
    cv2.putText(
        frame,
        instruction,
        (14, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 220, 120),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Box: {'PRESENT' if box_present else 'ABSENT'}  "
        f"change={presence_score:.3f}  motion={motion_ratio:.3f}  "
        "| R recalibrate | Q quit",
        (14, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def validate_args(args: argparse.Namespace) -> str | None:
    if not 0.0 <= args.conf <= 1.0:
        return "--conf must be between 0 and 1"
    if args.imgsz < 128:
        return "--imgsz must be at least 128"
    if args.calibration_frames < 1:
        return "--calibration-frames must be at least 1"
    if args.settle_frames < 1 or args.count_stable_frames < 1:
        return "--settle-frames and --count-stable-frames must be at least 1"
    if not 0.0 <= args.max_motion_ratio <= 1.0:
        return "--max-motion-ratio must be between 0 and 1"
    if not 0.0 <= args.absence_ratio < args.presence_ratio <= 1.0:
        return "thresholds must satisfy 0 <= absence-ratio < presence-ratio <= 1"
    if (
        args.pixel_threshold < 0
        or args.color_threshold < 0
        or args.strong_pixel_threshold < 0
        or args.motion_threshold < 0
    ):
        return "pixel, color, strong-pixel, and motion thresholds cannot be negative"
    if args.strong_pixel_threshold < args.pixel_threshold:
        return "strong-pixel-threshold must be at least pixel-threshold"
    return None


def main() -> int:
    args = parse_args()
    error = validate_args(args)
    if error:
        print(f"ERROR: {error}")
        return 2

    station_names = ["cans", "chili/stew"]
    stations = [
        Station(name, model_path.resolve(), target)
        for name, model_path, target in zip(station_names, args.models, args.targets)
    ]
    missing = [str(station.model_path) for station in stations if not station.model_path.is_file()]
    if missing:
        print("ERROR: Missing model weights:")
        for path in missing:
            print(f"  {path}")
        print("Place can.pt and chili-stew.pt beside smartpack.py, or use --models.")
        return 2

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Missing dependency.")
        print("Install with: python -m pip install ultralytics opencv-python")
        return 2

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        print(f"ERROR: Could not open camera index {args.camera}. Try --camera 1.")
        return 1
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    capture_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture_fps = camera.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera capture: {capture_width}x{capture_height} at "
        f"{capture_fps or 0:.1f} FPS | YOLO imgsz={args.imgsz} | "
        f"device={args.device if args.device is not None else 'auto'}"
    )

    detector = BoxPresenceDetector(cv2, args)
    if not detector.calibrate(camera):
        camera.release()
        cv2.destroyAllWindows()
        return 1

    active_index = 0
    saved_counts = [0] * len(stations)
    model = load_model(YOLO, stations[active_index])
    stable_count = StableCount(args.count_stable_frames)
    instantaneous_count = 0
    workflow_complete = False
    writer = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("ERROR: Could not read a camera frame.")
                break

            (
                box_present,
                box_event,
                mask,
                presence_score,
                motion_ratio,
                stable_frames,
            ) = detector.update(frame)

            if box_event == "entered" and not workflow_complete:
                stable_count.reset()
                instantaneous_count = 0
                reset_tracker(model)
                append_event(args.log, stations[active_index], "box_entered", 0)
                print(
                    f"Box entered station {active_index + 1}/{len(stations)}: "
                    f"{stations[active_index].name}"
                )

            annotated = frame.copy()
            if box_present and not workflow_complete:
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                annotated = result.plot()
                instantaneous_count = len(result.boxes) if result.boxes is not None else 0

                # Never replace the saved candidate while a hand or box is moving.
                # This prevents detections disappearing during removal from erasing
                # the last reliable count.
                if motion_ratio <= args.max_motion_ratio:
                    saved_counts[active_index] = stable_count.update(instantaneous_count)

                cv2.rectangle(
                    annotated,
                    (2, 2),
                    (annotated.shape[1] - 3, annotated.shape[0] - 3),
                    (60, 255, 60),
                    3,
                )

            if box_event == "exited" and not workflow_complete:
                final_count = stable_count.value
                saved_counts[active_index] = final_count
                current = stations[active_index]
                append_event(
                    args.log,
                    current,
                    "round_saved",
                    final_count,
                    f"target={current.target}",
                )
                print(f"Saved {current.name}: {final_count}/{current.target}")
                reset_tracker(model)
                instantaneous_count = 0

                if active_index < len(stations) - 1:
                    active_index += 1
                    model = load_model(YOLO, stations[active_index])
                    stable_count.reset()
                    print(
                        f"Ready for station {active_index + 1}/{len(stations)}: "
                        f"{stations[active_index].name}. Move the box back into view."
                    )
                else:
                    workflow_complete = True
                    passed = all(
                        count == station.target
                        for count, station in zip(saved_counts, stations)
                    )
                    append_event(
                        args.log,
                        current,
                        "workflow_complete",
                        final_count,
                        "PASS" if passed else "CHECK_COUNTS",
                    )
                    print(
                        "All station counts saved: "
                        + ", ".join(
                            f"{station.name}={saved_counts[index]}/{station.target}"
                            for index, station in enumerate(stations)
                        )
                    )
                    print("Result: PASS" if passed else "Result: CHECK COUNTS")

            draw_status(
                cv2,
                annotated,
                stations,
                saved_counts,
                active_index,
                box_present,
                instantaneous_count,
                stable_count.value,
                presence_score,
                motion_ratio,
                stable_frames,
                args.settle_frames,
                workflow_complete,
            )

            if args.save:
                if writer is None:
                    output = args.save.resolve()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    fps = capture_fps if capture_fps and capture_fps > 0 else 30.0
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {output}")
                    print(f"Recording annotated video to: {output}")
                writer.write(annotated)

            cv2.imshow("SmartPack", annotated)
            if args.show_mask:
                cv2.imshow("SmartPack box-presence mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                print("Recalibration requested. Remove the box, food, and hands.")
                time.sleep(1.0)
                if not detector.calibrate(camera):
                    break
                stable_count.reset()
                instantaneous_count = 0
                reset_tracker(model)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
