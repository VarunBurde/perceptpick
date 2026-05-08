"""TMP — build README/website media bundle from per-object videos and screenshots.

Produces in ``docs/media/``:

  * ``pickup_grid_<R>x<C>.mp4`` — labeled R objects × C grippers tiled video.
    Object names label the rows (left margin), gripper names label the
    columns (top margin), and a footer legend explains the green / red
    pose colours.
  * ``pickup_grid_<R>x<C>.gif`` — same grid downscaled, for inline README
    rendering (GitHub renders GIFs inline; MP4s only as download links).
  * ``pickup_grid_<R>x<C>_poster.png`` — single mid-clip frame.
  * ``grasps_screenshots.png`` — grid of the per-object screenshots with a
    category-color legend strip across the top.

Usage:
    pixi run python scripts/tmp_make_collages.py
    pixi run python scripts/tmp_make_collages.py --objects 5,8,13,21 \\
        --grippers ezgripper,franka,robotiq_2f_85,robotiq_2f_140,robotiq_3f,\\
                    kinova_3f,sawyer,wsg_32,wsg_50
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from perceptpick.configs import YCB_OBJECTS
from perceptpick.grippers import GRIPPER_REGISTRY, all_grippers
from perceptpick.paths import resolve_paths

_log = logging.getLogger(__name__)

DEFAULT_OBJECT_IDS = [5, 8, 13, 21]  # 4 diverse objects
# The paper's 9 grippers (perceptpick.grippers.all_grippers) — what
# scripts/02_grasp_sweep.py runs by default. The full GRIPPER_REGISTRY also
# includes barrett/barrett_2f/rg2 which the canonical sweep does NOT cover.
_CLS_TO_NAME = {cls: name for name, cls in GRIPPER_REGISTRY.items()}
DEFAULT_GRIPPERS = sorted(_CLS_TO_NAME[c] for c in all_grippers if c in _CLS_TO_NAME)

# Category colours used by tmp_screenshot_grasps.py.
CATEGORY_LEGEND = [
    ("successful",       "#1ad81a"),
    ("collision_target", "#ffd900"),
    ("no_contact",       "#1a4dff"),
    ("slipped",          "#ff1ad8"),
    ("error",            "#8c8c8c"),
]

# macOS system fonts that ffmpeg's drawtext can read. First one that exists wins.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _pick_font() -> str | None:
    for c in _FONT_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def _resolve_video(videos_dir: Path, obj: str, gripper: str) -> Path | None:
    matches = sorted(videos_dir.glob(f"{obj}__{gripper}__*.mp4"))
    return matches[0] if matches else None


def _parse_int_list(s: str | None, default: list[int]) -> list[int]:
    return [int(x) for x in s.split(",")] if s else list(default)


def _parse_str_list(s: str | None, default: list[str]) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else list(default)


def _render_label_overlay(
    canvas_w: int, canvas_h: int,
    label_top: int, label_left: int, legend_h: int,
    grid_w: int, grid_h: int,
    object_names: list[str], grippers: list[str],
    tile_w: int, tile_h: int,
    font_path: str, label_font_size: int, legend_font_size: int,
    out_path: Path,
) -> None:
    """Render the static labels + legend strip as an RGBA PNG. The grid
    region is fully transparent so the video shows through; margins are
    opaque white with black text. Separate from ffmpeg because Homebrew's
    default ffmpeg often ships without ``drawtext`` (libfreetype not
    enabled).
    """
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    # Cut a transparent hole where the video grid sits.
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [label_left, label_top, label_left + grid_w, label_top + grid_h],
        fill=(0, 0, 0, 0),
    )
    label_font = ImageFont.truetype(font_path, label_font_size)
    legend_font = ImageFont.truetype(font_path, legend_font_size)

    # Column labels (gripper names) along the top
    for c, gr in enumerate(grippers):
        bbox = draw.textbbox((0, 0), gr, font=label_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = label_left + c * tile_w + (tile_w - tw) // 2
        y = (label_top - th) // 2
        draw.text((x, y), gr, fill=(0, 0, 0, 255), font=label_font)

    # Row labels (object names) along the left
    for r, obj in enumerate(object_names):
        text = _short_name(obj)
        bbox = draw.textbbox((0, 0), text, font=label_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (label_left - tw) // 2
        y = label_top + r * tile_h + (tile_h - th) // 2
        draw.text((x, y), text, fill=(0, 0, 0, 255), font=label_font)

    # Footer legend
    legend_text = (
        "Green: GT pose   |   Red: pose-estimator pose (BakedSDF + FoundationPose)"
        "   |   RGB axes at world origin"
    )
    bbox = draw.textbbox((0, 0), legend_text, font=legend_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (canvas_w - tw) // 2
    y = label_top + grid_h + (legend_h - th) // 2
    draw.text((x, y), legend_text, fill=(0, 0, 0, 255), font=legend_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def build_video_grid(
    videos_dir: Path,
    out_mp4: Path,
    object_names: list[str],
    grippers: list[str],
    tile_w: int,
    tile_h: int,
    duration_sec: float,
    fps: int,
    *,
    label_top: int,
    label_left: int,
    legend_h: int,
    font: str,
    label_font_size: int,
    legend_font_size: int,
) -> bool:
    cols = len(grippers)
    rows = len(object_names)

    inputs: list[Path] = []
    missing: list[str] = []
    for obj in object_names:
        for gr in grippers:
            v = _resolve_video(videos_dir, obj, gr)
            if v is None:
                missing.append(f"{obj} × {gr}")
            else:
                inputs.append(v)
    if missing:
        _log.error("missing %d video(s):\n  %s", len(missing), "\n  ".join(missing))
        return False

    grid_w = cols * tile_w
    grid_h = rows * tile_h
    canvas_w = label_left + grid_w
    canvas_h = label_top + grid_h + legend_h

    overlay_png = out_mp4.with_name(out_mp4.stem + "__labels.png")
    _render_label_overlay(
        canvas_w, canvas_h, label_top, label_left, legend_h,
        grid_w, grid_h, object_names, grippers, tile_w, tile_h,
        font, label_font_size, legend_font_size, overlay_png,
    )

    # Build ffmpeg command: tile videos via xstack, pad to canvas, overlay
    # the static label PNG on top.
    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    for v in inputs:
        cmd += ["-stream_loop", "-1", "-t", f"{duration_sec}", "-i", str(v)]
    cmd += ["-loop", "1", "-i", str(overlay_png)]
    overlay_idx = len(inputs)

    parts: list[str] = []
    for i in range(len(inputs)):
        parts.append(f"[{i}:v]scale={tile_w}:{tile_h},setpts=PTS-STARTPTS,fps={fps}[v{i}]")
    layout_tokens = []
    for r in range(rows):
        for c in range(cols):
            layout_tokens.append(f"{c * tile_w}_{r * tile_h}")
    layout = "|".join(layout_tokens)
    chain = "".join(f"[v{i}]" for i in range(len(inputs)))
    parts.append(f"{chain}xstack=inputs={len(inputs)}:layout={layout}:fill=white[grid]")
    parts.append(
        f"[grid]pad={canvas_w}:{canvas_h}:{label_left}:{label_top}:white[padded]"
    )
    parts.append(f"[padded][{overlay_idx}:v]overlay=0:0:format=auto[out]")

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[out]",
        "-t", f"{duration_sec}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-r", str(fps),
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    overlay_png.unlink(missing_ok=True)
    if res.returncode != 0:
        _log.error("ffmpeg xstack failed:\n%s", res.stderr.strip())
        return False
    print(f"  wrote {out_mp4} ({canvas_w}×{canvas_h}, {duration_sec}s @ {fps}fps)")
    return True


def _escape(s: str) -> str:
    """Escape characters that ffmpeg drawtext interprets specially."""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("|", "\\|")


def _short_name(obj_name: str) -> str:
    """`006_mustard_bottle` → `mustard_bottle` for display."""
    parts = obj_name.split("_", 1)
    return parts[1] if len(parts) > 1 and parts[0].isdigit() else obj_name


def make_gif(in_mp4: Path, out_gif: Path, width: int, fps: int) -> bool:
    """Palette-quality GIF from the grid mp4 (GitHub renders GIFs inline)."""
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    palette = out_gif.with_suffix(".palette.png")
    # max_colors=256 with stats_mode=full gives a richer palette for a
    # mostly-static scene with small moving subjects (the grippers/objects).
    pal_cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(in_mp4),
        "-vf",
        f"fps={fps},scale={width}:-2:flags=lanczos,"
        f"palettegen=stats_mode=full:max_colors=256",
        str(palette),
    ]
    if subprocess.run(pal_cmd, capture_output=True, text=True).returncode != 0:
        return False
    # Floyd-Steinberg dithering preserves the green/red gradients better than
    # ordered-bayer at this resolution.
    gif_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(in_mp4), "-i", str(palette),
        "-filter_complex",
        f"[0:v]fps={fps},scale={width}:-2:flags=lanczos[x];"
        f"[x][1:v]paletteuse=dither=floyd_steinberg:diff_mode=rectangle",
        str(out_gif),
    ]
    rc = subprocess.run(gif_cmd, capture_output=True, text=True).returncode
    palette.unlink(missing_ok=True)
    if rc != 0:
        return False
    print(f"  wrote {out_gif} ({width}px wide @ {fps}fps)")
    return True


def make_poster(in_mp4: Path, out_png: Path, time_sec: float) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{time_sec}", "-i", str(in_mp4),
           "-frames:v", "1", "-q:v", "2", str(out_png)]
    if subprocess.run(cmd, capture_output=True, text=True).returncode != 0:
        return False
    print(f"  wrote {out_png}")
    return True


def build_screenshot_grid(
    screens_dir: Path, out_png: Path, cols: int, rows: int,
    tile_w: int, tile_h: int,
) -> bool:
    """Tile screenshots into a grid + add a category-color legend across the top."""
    from PIL import Image, ImageDraw, ImageFont

    paths = sorted(screens_dir.glob("*.png"))
    if not paths:
        _log.error("no PNGs in %s", screens_dir)
        return False
    if len(paths) > cols * rows:
        paths = paths[: cols * rows]

    # Legend strip dimensions
    legend_h = 70
    swatch_size = 28
    spacing = 18
    title_size = 24

    # Build the grid first
    grid_w = cols * tile_w
    grid_h = rows * tile_h
    canvas = Image.new("RGB", (grid_w, grid_h + legend_h), "white")

    # Try to load a font; fall back to default if none works
    font = None
    for fp in _FONT_CANDIDATES:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, title_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)

    # Compute legend layout: render swatches + labels left-to-right, centre on canvas
    # First, measure total width
    items = []
    for label, hex_color in CATEGORY_LEGEND:
        rgb = _hex_to_rgb(hex_color)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        items.append((label, rgb, text_w, text_h))
    total_w = sum(swatch_size + 8 + iw for _, _, iw, _ in items) + spacing * (len(items) - 1)
    x = (grid_w - total_w) // 2
    y = (legend_h - swatch_size) // 2

    for label, rgb, text_w, text_h in items:
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=rgb,
                       outline=(0, 0, 0), width=1)
        text_y = y + (swatch_size - text_h) // 2 - 2
        draw.text((x + swatch_size + 8, text_y), label, fill="black", font=font)
        x += swatch_size + 8 + text_w + spacing

    # Now paste the screenshot tiles below the legend
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        img.thumbnail((tile_w, tile_h), Image.LANCZOS)
        cell_x = (i % cols) * tile_w + (tile_w - img.width) // 2
        cell_y = legend_h + (i // cols) * tile_h + (tile_h - img.height) // 2
        canvas.paste(img, (cell_x, cell_y))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, optimize=True)
    print(f"  wrote {out_png} ({cols}×{rows} tiles + legend, {grid_w}×{grid_h + legend_h})")
    return True


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gt-mesh", default="GT")
    parser.add_argument("--est-mesh", default="BakedSDF")
    parser.add_argument("--objects", default=None,
                        help=f"comma-separated YCB ids (rows; default: {DEFAULT_OBJECT_IDS})")
    parser.add_argument("--grippers", default=None,
                        help=f"comma-separated gripper names (cols; default: all 9 in registry)")
    parser.add_argument("--tile-w", type=int, default=320)
    parser.add_argument("--tile-h", type=int, default=240)
    parser.add_argument("--duration", type=float, default=12.0,
                        help="seconds; shorter videos are looped to fill")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--gif-width", type=int, default=1600)
    parser.add_argument("--gif-fps", type=int, default=18)
    parser.add_argument("--poster-time", type=float, default=4.0)
    parser.add_argument("--label-top", type=int, default=56)
    parser.add_argument("--label-left", type=int, default=170)
    parser.add_argument("--legend-h", type=int, default=52)
    parser.add_argument("--label-font-size", type=int, default=24)
    parser.add_argument("--legend-font-size", type=int, default=20)
    parser.add_argument("--out-dir", default=None,
                        help="default: docs/media/ at project root")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = resolve_paths(output_root=args.output_root)

    object_ids = _parse_int_list(args.objects, DEFAULT_OBJECT_IDS)
    object_names = [YCB_OBJECTS[i] for i in object_ids if i in YCB_OBJECTS]
    grippers = _parse_str_list(args.grippers, DEFAULT_GRIPPERS)

    videos_dir = paths.output_root / "visualization" / "videos" / f"{args.gt_mesh}_vs_{args.est_mesh}"
    screens_dir = paths.output_root / "visualization" / "screenshots" / args.gt_mesh
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "docs" / "media"
    )
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not on PATH; install it first")
        return
    font = _pick_font()
    if font is None:
        print("no system TTF found in expected paths; cannot render labels")
        return

    rows = len(object_names)
    cols = len(grippers)
    tag = f"{rows}x{cols}"
    grid_mp4 = out_dir / f"pickup_grid_{tag}.mp4"

    print(f"video grid → {grid_mp4}  ({rows} objects × {cols} grippers)")
    if not build_video_grid(
        videos_dir, grid_mp4, object_names, grippers,
        args.tile_w, args.tile_h, args.duration, args.fps,
        label_top=args.label_top, label_left=args.label_left,
        legend_h=args.legend_h, font=font,
        label_font_size=args.label_font_size,
        legend_font_size=args.legend_font_size,
    ):
        return

    print(f"GIF preview → {out_dir}/pickup_grid_{tag}.gif")
    make_gif(grid_mp4, out_dir / f"pickup_grid_{tag}.gif", args.gif_width, args.gif_fps)

    print(f"poster frame → {out_dir}/pickup_grid_{tag}_poster.png")
    make_poster(grid_mp4, out_dir / f"pickup_grid_{tag}_poster.png", args.poster_time)

    if screens_dir.is_dir():
        print(f"screenshot collage → {out_dir}/grasps_screenshots.png")
        build_screenshot_grid(screens_dir, out_dir / "grasps_screenshots.png",
                              cols=4, rows=2, tile_w=640, tile_h=480)

    print(f"\ndone. README assets in {out_dir}/")


if __name__ == "__main__":
    main()
