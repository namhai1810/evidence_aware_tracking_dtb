#!/usr/bin/env python3
"""Generate the HIVE v2 report figures and a styled PDF without third-party Python packages."""

from __future__ import annotations

import ctypes
import ctypes.util
import html
import json
import math
import re
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CODE_DIR / "outputs"
ASSET_DIR = OUTPUT_DIR / "report_assets"
REPORT_MD = OUTPUT_DIR / "23042026.md"
REPORT_PDF = OUTPUT_DIR / "23042026.pdf"
TRAIN_METRICS = CODE_DIR / "hr_finding/runs/hive_history_v2_seed42/metrics.json"

CAIRO_FORMAT_ARGB32 = 0
CAIRO_FONT_SLANT_NORMAL = 0
CAIRO_FONT_SLANT_ITALIC = 1
CAIRO_FONT_WEIGHT_NORMAL = 0
CAIRO_FONT_WEIGHT_BOLD = 1


class TextExtents(ctypes.Structure):
    _fields_ = [
        ("x_bearing", ctypes.c_double),
        ("y_bearing", ctypes.c_double),
        ("width", ctypes.c_double),
        ("height", ctypes.c_double),
        ("x_advance", ctypes.c_double),
        ("y_advance", ctypes.c_double),
    ]


class Cairo:
    def __init__(self) -> None:
        library = ctypes.util.find_library("cairo")
        if not library:
            raise RuntimeError("libcairo was not found")
        self.lib = ctypes.CDLL(library)
        pointer = ctypes.c_void_p
        self.lib.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.cairo_image_surface_create.restype = pointer
        self.lib.cairo_image_surface_create_from_png.argtypes = [ctypes.c_char_p]
        self.lib.cairo_image_surface_create_from_png.restype = pointer
        self.lib.cairo_image_surface_get_width.argtypes = [pointer]
        self.lib.cairo_image_surface_get_width.restype = ctypes.c_int
        self.lib.cairo_image_surface_get_height.argtypes = [pointer]
        self.lib.cairo_image_surface_get_height.restype = ctypes.c_int
        self.lib.cairo_surface_write_to_png.argtypes = [pointer, ctypes.c_char_p]
        self.lib.cairo_surface_write_to_png.restype = ctypes.c_int
        self.lib.cairo_pdf_surface_create.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
        self.lib.cairo_pdf_surface_create.restype = pointer
        self.lib.cairo_surface_destroy.argtypes = [pointer]
        self.lib.cairo_surface_finish.argtypes = [pointer]
        self.lib.cairo_create.argtypes = [pointer]
        self.lib.cairo_create.restype = pointer
        self.lib.cairo_destroy.argtypes = [pointer]
        for name in ("move_to", "line_to", "translate", "scale"):
            getattr(self.lib, f"cairo_{name}").argtypes = [pointer, ctypes.c_double, ctypes.c_double]
        self.lib.cairo_rectangle.argtypes = [pointer, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.cairo_arc.argtypes = [pointer, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.cairo_close_path.argtypes = [pointer]
        self.lib.cairo_set_source_rgb.argtypes = [pointer, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.cairo_set_line_width.argtypes = [pointer, ctypes.c_double]
        self.lib.cairo_fill.argtypes = [pointer]
        self.lib.cairo_fill_preserve.argtypes = [pointer]
        self.lib.cairo_stroke.argtypes = [pointer]
        self.lib.cairo_paint.argtypes = [pointer]
        self.lib.cairo_save.argtypes = [pointer]
        self.lib.cairo_restore.argtypes = [pointer]
        self.lib.cairo_show_page.argtypes = [pointer]
        self.lib.cairo_select_font_face.argtypes = [pointer, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.lib.cairo_set_font_size.argtypes = [pointer, ctypes.c_double]
        self.lib.cairo_show_text.argtypes = [pointer, ctypes.c_char_p]
        self.lib.cairo_text_extents.argtypes = [pointer, ctypes.c_char_p, ctypes.POINTER(TextExtents)]
        self.lib.cairo_set_source_surface.argtypes = [pointer, pointer, ctypes.c_double, ctypes.c_double]

    def rgb(self, context: int, color: str) -> None:
        value = color.lstrip("#")
        red, green, blue = (int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))
        self.lib.cairo_set_source_rgb(context, red, green, blue)

    def font(self, context: int, size: float, bold: bool = False, italic: bool = False) -> None:
        self.lib.cairo_select_font_face(
            context,
            b"DejaVu Sans",
            CAIRO_FONT_SLANT_ITALIC if italic else CAIRO_FONT_SLANT_NORMAL,
            CAIRO_FONT_WEIGHT_BOLD if bold else CAIRO_FONT_WEIGHT_NORMAL,
        )
        self.lib.cairo_set_font_size(context, size)

    def text_width(self, context: int, value: str) -> float:
        extents = TextExtents()
        self.lib.cairo_text_extents(context, value.encode("utf-8"), ctypes.byref(extents))
        return extents.x_advance

    def text(self, context: int, x: float, y: float, value: str, color: str = "#0f172a") -> None:
        self.rgb(context, color)
        self.lib.cairo_move_to(context, x, y)
        self.lib.cairo_show_text(context, value.encode("utf-8"))


CAIRO = Cairo()


def rounded_rect(context: int, x: float, y: float, width: float, height: float, radius: float) -> None:
    radius = min(radius, width / 2, height / 2)
    CAIRO.lib.cairo_move_to(context, x + radius, y)
    CAIRO.lib.cairo_line_to(context, x + width - radius, y)
    CAIRO.lib.cairo_arc(context, x + width - radius, y + radius, radius, -math.pi / 2, 0)
    CAIRO.lib.cairo_line_to(context, x + width, y + height - radius)
    CAIRO.lib.cairo_arc(context, x + width - radius, y + height - radius, radius, 0, math.pi / 2)
    CAIRO.lib.cairo_line_to(context, x + radius, y + height)
    CAIRO.lib.cairo_arc(context, x + radius, y + height - radius, radius, math.pi / 2, math.pi)
    CAIRO.lib.cairo_line_to(context, x, y + radius)
    CAIRO.lib.cairo_arc(context, x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
    CAIRO.lib.cairo_close_path(context)


def fill_box(context: int, x: float, y: float, width: float, height: float, fill: str, stroke: str | None = None) -> None:
    rounded_rect(context, x, y, width, height, 18)
    CAIRO.rgb(context, fill)
    if stroke:
        CAIRO.lib.cairo_fill_preserve(context)
        CAIRO.rgb(context, stroke)
        CAIRO.lib.cairo_set_line_width(context, 2)
        CAIRO.lib.cairo_stroke(context)
    else:
        CAIRO.lib.cairo_fill(context)


def centered_text(context: int, center_x: float, y: float, value: str, size: float, color: str, bold: bool = False) -> None:
    CAIRO.font(context, size, bold=bold)
    CAIRO.text(context, center_x - CAIRO.text_width(context, value) / 2, y, value, color)


def draw_summary_png(path: Path) -> None:
    width, height = 1800, 1100
    surface = CAIRO.lib.cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height)
    context = CAIRO.lib.cairo_create(surface)
    CAIRO.rgb(context, "#0b1220")
    CAIRO.lib.cairo_paint(context)
    CAIRO.font(context, 48, bold=True)
    CAIRO.text(context, 70, 75, "HIVE v2 — LR-only historical frame ranking", "#f8fafc")
    CAIRO.font(context, 23)
    CAIRO.text(context, 70, 115, "17 videos · 925 switch events · video-disjoint test · no HSV model input", "#94a3b8")

    panels = [(55, 155), (920, 155), (55, 610), (920, 610)]
    for x, y in panels:
        fill_box(context, x, y, 825, 405, "#111c30", "#2b3b56")

    CAIRO.font(context, 28, bold=True)
    CAIRO.text(context, 90, 205, "A. LR-only candidate construction", "#e2e8f0")
    stages = [
        ("Known LR switches", "925"),
        ("Previous history", "T−150…T−1"),
        ("LR candidates", "≤20/event"),
        ("Frozen pool", "18,355 pairs"),
    ]
    for index, (label, value) in enumerate(stages):
        x = 90 + (index % 2) * 380
        y = 245 + (index // 2) * 135
        fill_box(context, x, y, 340, 105, "#1a2942", "#38506f")
        centered_text(context, x + 170, y + 38, label, 19, "#a9b7ca")
        centered_text(context, x + 170, y + 78, value, 31, "#5eead4", bold=True)
    CAIRO.font(context, 17)
    CAIRO.text(context, 90, 520, "Pool is frozen before Hungarian GT labels are joined (IoU ≥ 0.50).", "#94a3b8")

    CAIRO.font(context, 28, bold=True)
    CAIRO.text(context, 955, 205, "B. Video-disjoint data split", "#e2e8f0")
    split_data = [
        ("TRAIN", "10 videos", "757 events", "14,997 pairs", "#0e7490"),
        ("VALIDATION", "3 videos", "76 events", "1,518 pairs", "#b7791f"),
        ("TEST", "4 videos", "92 events", "1,840 pairs", "#6d5bc6"),
    ]
    for index, (name, videos, events, pairs, color) in enumerate(split_data):
        x = 955 + index * 255
        fill_box(context, x, 250, 225, 190, "#18263c", color)
        centered_text(context, x + 112, 286, name, 19, "#e2e8f0", bold=True)
        centered_text(context, x + 112, 330, videos, 26, "#f8fafc", bold=True)
        centered_text(context, x + 112, 368, events, 20, "#cbd5e1")
        centered_text(context, x + 112, 405, pairs, 20, "#cbd5e1")
    fill_box(context, 955, 465, 735, 60, "#102b27", "#247a69")
    centered_text(context, 1322, 503, "PASS · zero shared videos, events, or source frames", 20, "#5eead4", bold=True)

    CAIRO.font(context, 28, bold=True)
    CAIRO.text(context, 90, 660, "C. HIVE v2 model", "#e2e8f0")
    model_stages = [
        ("LR evidence", "old · target · candidates"),
        ("Shared tiny CNN", "64-D appearance"),
        ("Relations + time", "competing hypothesis"),
        ("Utility score", "rank each frame"),
    ]
    for index, (name, detail) in enumerate(model_stages):
        x = 90 + index * 188
        fill_box(context, x, 715, 165, 145, "#192942", "#38506f")
        centered_text(context, x + 82, 764, name, 16.5, "#f8fafc", bold=True)
        centered_text(context, x + 82, 811, detail, 12.5, "#a9b7ca")
        if index < len(model_stages) - 1:
            CAIRO.rgb(context, "#64748b")
            CAIRO.lib.cairo_set_line_width(context, 4)
            CAIRO.lib.cairo_move_to(context, x + 168, 785)
            CAIRO.lib.cairo_line_to(context, x + 186, 785)
            CAIRO.lib.cairo_stroke(context)
    fill_box(context, 90, 890, 750, 90, "#162d28", "#2d8c75")
    centered_text(context, 465, 925, "146,251 parameters · BCE + 0.5 × RankNet", 20, "#cbd5e1")
    centered_text(context, 465, 958, "0.815 ms/event model latency", 23, "#5eead4", bold=True)

    CAIRO.font(context, 28, bold=True)
    CAIRO.text(context, 955, 660, "D. Held-out test Success@1", "#e2e8f0")
    methods = [
        ("Nearest frame", 27.17, "25/92", "#64748b"),
        ("LR proposal order", 52.17, "48/92", "#f59e0b"),
        ("HIVE v2", 63.04, "58/92", "#2dd4bf"),
        ("GT pool oracle", 86.96, "80/92", "#8b5cf6"),
    ]
    for index, (name, value, count, color) in enumerate(methods):
        y = 715 + index * 62
        CAIRO.font(context, 18, bold=(name == "HIVE v2"))
        CAIRO.text(context, 955, y + 25, name, "#e2e8f0")
        CAIRO.rgb(context, "#24344e")
        CAIRO.lib.cairo_rectangle(context, 1160, y, 470, 34)
        CAIRO.lib.cairo_fill(context)
        CAIRO.rgb(context, color)
        CAIRO.lib.cairo_rectangle(context, 1160, y, 470 * value / 100, 34)
        CAIRO.lib.cairo_fill(context)
        value_text = f"{value:.2f}% · {count}"
        CAIRO.font(context, 18, bold=True)
        CAIRO.text(
            context,
            1690 - CAIRO.text_width(context, value_text),
            y + 25,
            value_text,
            "#f8fafc",
        )
    fill_box(context, 955, 980, 735, 70, "#1a2942", "#38506f")
    centered_text(context, 1322, 1010, "MRR 72.19% · NDCG@5 68.97%", 22, "#f8fafc", bold=True)
    centered_text(context, 1322, 1038, "22 retrievable top-1 failures remain", 17, "#a9b7ca")

    CAIRO.lib.cairo_surface_write_to_png(surface, str(path).encode())
    CAIRO.lib.cairo_destroy(context)
    CAIRO.lib.cairo_surface_destroy(surface)


def draw_diagnostics_png(path: Path) -> None:
    metrics = json.loads(TRAIN_METRICS.read_text(encoding="utf-8"))
    width, height = 1800, 1000
    surface = CAIRO.lib.cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height)
    context = CAIRO.lib.cairo_create(surface)
    CAIRO.rgb(context, "#f1f5f9")
    CAIRO.lib.cairo_paint(context)
    CAIRO.font(context, 44, bold=True)
    CAIRO.text(context, 65, 70, "Failure profile and HIVE v2 training diagnostics", "#0f172a")
    CAIRO.font(context, 21)
    CAIRO.text(context, 65, 108, "Counts overlap across failure categories · best checkpoint selected only on validation", "#64748b")

    fill_box(context, 50, 145, 800, 380, "#ffffff", "#cbd5e1")
    CAIRO.font(context, 27, bold=True)
    CAIRO.text(context, 85, 190, "A. 512×120 YOLO11l + ByteTrack incorrect frames", "#0f172a")
    failure_rows = [
        ("Any incorrect frame", 8195, "#334155"),
        ("At least one FN", 8030, "#ef4444"),
        ("At least one FP", 3708, "#f59e0b"),
        ("At least one ID switch", 1130, "#8b5cf6"),
    ]
    maximum = 8195
    for index, (label, value, color) in enumerate(failure_rows):
        y = 225 + index * 68
        CAIRO.font(context, 18)
        CAIRO.text(context, 85, y + 24, label, "#334155")
        CAIRO.rgb(context, "#e2e8f0")
        CAIRO.lib.cairo_rectangle(context, 320, y, 430, 32)
        CAIRO.lib.cairo_fill(context)
        CAIRO.rgb(context, color)
        CAIRO.lib.cairo_rectangle(context, 320, y, 430 * value / maximum, 32)
        CAIRO.lib.cairo_fill(context)
        CAIRO.font(context, 18, bold=True)
        CAIRO.text(context, 765, y + 24, f"{value:,}", "#0f172a")
    CAIRO.font(context, 15, italic=True)
    CAIRO.text(context, 85, 492, "A frame may contain FN, FP, and an ID switch simultaneously.", "#64748b")

    fill_box(context, 875, 145, 875, 380, "#ffffff", "#cbd5e1")
    CAIRO.font(context, 27, bold=True)
    CAIRO.text(context, 910, 190, "B. Optimization and validation behavior", "#0f172a")
    plot_x, plot_y, plot_w, plot_h = 950, 230, 720, 225
    CAIRO.rgb(context, "#e2e8f0")
    for index in range(5):
        y = plot_y + index * plot_h / 4
        CAIRO.lib.cairo_move_to(context, plot_x, y)
        CAIRO.lib.cairo_line_to(context, plot_x + plot_w, y)
        CAIRO.lib.cairo_stroke(context)
    max_loss = max(max(item["train_loss_total"], item["val_loss_total"]) for item in metrics) * 1.05
    colors = [("train_loss_total", "#0ea5e9"), ("val_loss_total", "#ef4444")]
    for key, color in colors:
        CAIRO.rgb(context, color)
        CAIRO.lib.cairo_set_line_width(context, 5)
        for index, item in enumerate(metrics):
            x = plot_x + index * plot_w / (len(metrics) - 1)
            y = plot_y + plot_h - item[key] / max_loss * plot_h
            if index == 0:
                CAIRO.lib.cairo_move_to(context, x, y)
            else:
                CAIRO.lib.cairo_line_to(context, x, y)
        CAIRO.lib.cairo_stroke(context)
    success = [item["val_success_at_1"] for item in metrics]
    CAIRO.rgb(context, "#10b981")
    CAIRO.lib.cairo_set_line_width(context, 5)
    for index, value in enumerate(success):
        x = plot_x + index * plot_w / (len(metrics) - 1)
        y = plot_y + plot_h - value * plot_h
        if index == 0:
            CAIRO.lib.cairo_move_to(context, x, y)
        else:
            CAIRO.lib.cairo_line_to(context, x, y)
    CAIRO.lib.cairo_stroke(context)
    best_index = max(range(len(success)), key=success.__getitem__)
    best_x = plot_x + best_index * plot_w / (len(metrics) - 1)
    CAIRO.rgb(context, "#0f172a")
    CAIRO.lib.cairo_set_line_width(context, 2)
    CAIRO.lib.cairo_move_to(context, best_x, plot_y)
    CAIRO.lib.cairo_line_to(context, best_x, plot_y + plot_h)
    CAIRO.lib.cairo_stroke(context)
    CAIRO.font(context, 15, bold=True)
    CAIRO.text(context, best_x + 8, plot_y + 20, "best epoch 3", "#0f172a")
    legend = [("train loss", "#0ea5e9"), ("validation loss", "#ef4444"), ("validation Success@1", "#10b981")]
    for index, (label, color) in enumerate(legend):
        x = 950 + index * 225
        CAIRO.rgb(context, color)
        CAIRO.lib.cairo_rectangle(context, x, 474, 24, 8)
        CAIRO.lib.cairo_fill(context)
        CAIRO.font(context, 14)
        CAIRO.text(context, x + 32, 486, label, "#475569")

    fill_box(context, 50, 555, 800, 380, "#ffffff", "#cbd5e1")
    CAIRO.font(context, 27, bold=True)
    CAIRO.text(context, 85, 600, "C. Held-out retrieval funnel", "#0f172a")
    funnel = [
        ("Test switch events", 92, "#334155"),
        ("At least one positive in pool", 80, "#0ea5e9"),
        ("HIVE correct at top-1", 58, "#10b981"),
        ("Retrievable but top-1 wrong", 22, "#ef4444"),
    ]
    for index, (label, value, color) in enumerate(funnel):
        y = 640 + index * 65
        CAIRO.font(context, 18)
        CAIRO.text(context, 85, y + 25, label, "#334155")
        CAIRO.rgb(context, color)
        CAIRO.lib.cairo_rectangle(context, 360, y, 390 * value / 92, 34)
        CAIRO.lib.cairo_fill(context)
        CAIRO.font(context, 18, bold=True)
        CAIRO.text(context, 765, y + 25, str(value), "#0f172a")
    CAIRO.font(context, 15, italic=True)
    CAIRO.text(context, 85, 910, "The 12 events without positives are candidate-pool failures, not ranking errors.", "#64748b")

    fill_box(context, 875, 555, 875, 380, "#ffffff", "#cbd5e1")
    CAIRO.font(context, 27, bold=True)
    CAIRO.text(context, 910, 600, "D. Current experimental boundary", "#0f172a")
    scope = [
        ("✓", "Implemented and measured", "LR candidate mining; video-disjoint ranking; latency", "#047857"),
        ("✓", "Supervision isolated", "GT is joined only after the candidate pool is frozen", "#047857"),
        ("—", "Deferred", "Online switch trigger and non-switch events", "#b7791f"),
        ("—", "Deferred", "Identity merge, tracker replay, post-merge IDSW/IDF1/MOTA", "#b7791f"),
    ]
    for index, (symbol, title, detail, color) in enumerate(scope):
        y = 650 + index * 67
        fill_box(context, 910, y - 29, 58, 50, "#f8fafc", color)
        centered_text(context, 939, y + 5, symbol, 23, color, bold=True)
        CAIRO.font(context, 18, bold=True)
        CAIRO.text(context, 990, y - 5, title, color)
        CAIRO.font(context, 15)
        CAIRO.text(context, 990, y + 19, detail, "#475569")

    CAIRO.lib.cairo_surface_write_to_png(surface, str(path).encode())
    CAIRO.lib.cairo_destroy(context)
    CAIRO.lib.cairo_surface_destroy(surface)


def clean_markdown(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    return value.strip()


class PdfRenderer:
    PAGE_WIDTH = 842.0
    PAGE_HEIGHT = 595.0
    MARGIN_X = 36.0
    TOP = 38.0
    BOTTOM = 28.0

    def __init__(self, output: Path) -> None:
        self.surface = CAIRO.lib.cairo_pdf_surface_create(
            str(output).encode(), self.PAGE_WIDTH, self.PAGE_HEIGHT
        )
        self.context = CAIRO.lib.cairo_create(self.surface)
        self.y = self.TOP
        self.page = 1
        self._page_background()

    @property
    def content_width(self) -> float:
        return self.PAGE_WIDTH - 2 * self.MARGIN_X

    def _page_background(self) -> None:
        CAIRO.rgb(self.context, "#ffffff")
        CAIRO.lib.cairo_paint(self.context)
        CAIRO.rgb(self.context, "#0f766e")
        CAIRO.lib.cairo_rectangle(self.context, 0, 0, self.PAGE_WIDTH, 8)
        CAIRO.lib.cairo_fill(self.context)

    def _footer(self) -> None:
        CAIRO.font(self.context, 7.5)
        CAIRO.text(self.context, self.MARGIN_X, self.PAGE_HEIGHT - 15, "Evidence-Aware Tracking · HIVE v2 · pre-HSV-input", "#64748b")
        page_text = f"Page {self.page}"
        CAIRO.text(
            self.context,
            self.PAGE_WIDTH - self.MARGIN_X - CAIRO.text_width(self.context, page_text),
            self.PAGE_HEIGHT - 15,
            page_text,
            "#64748b",
        )

    def new_page(self) -> None:
        self._footer()
        CAIRO.lib.cairo_show_page(self.context)
        self.page += 1
        self.y = self.TOP
        self._page_background()

    def ensure(self, required: float) -> None:
        if self.y + required > self.PAGE_HEIGHT - self.BOTTOM:
            self.new_page()

    def wrap(self, text: str, width: float, size: float, bold: bool = False, italic: bool = False) -> list[str]:
        CAIRO.font(self.context, size, bold=bold, italic=italic)
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if CAIRO.text_width(self.context, candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def paragraph(
        self,
        text: str,
        size: float = 8.8,
        color: str = "#1e293b",
        bold: bool = False,
        italic: bool = False,
        indent: float = 0,
        before: float = 2,
        after: float = 4,
    ) -> None:
        text = clean_markdown(text)
        lines = self.wrap(text, self.content_width - indent, size, bold, italic)
        line_height = size * 1.32
        self.ensure(before + len(lines) * line_height + after)
        self.y += before
        CAIRO.font(self.context, size, bold=bold, italic=italic)
        for line in lines:
            CAIRO.text(self.context, self.MARGIN_X + indent, self.y + size, line, color)
            self.y += line_height
        self.y += after

    def heading(self, level: int, text: str) -> None:
        sizes = {1: 22, 2: 15, 3: 11.5}
        size = sizes[level]
        # Keep a heading with at least the first paragraph/list item instead of
        # leaving an isolated heading at the bottom of a page.
        needed = size * 1.8 + (42 if level == 1 else 34)
        self.ensure(needed)
        self.y += 8 if level == 1 else 5
        if level == 2:
            CAIRO.rgb(self.context, "#ccfbf1")
            CAIRO.lib.cairo_rectangle(self.context, self.MARGIN_X - 7, self.y - 3, self.content_width + 14, size * 1.65)
            CAIRO.lib.cairo_fill(self.context)
        CAIRO.font(self.context, size, bold=True)
        CAIRO.text(self.context, self.MARGIN_X, self.y + size, clean_markdown(text), "#0f172a" if level > 1 else "#0f766e")
        self.y += size * 1.55 + (6 if level == 1 else 4)

    def bullet(self, text: str, ordered: str | None = None) -> None:
        marker = f"{ordered}." if ordered else "•"
        CAIRO.font(self.context, 8.8, bold=bool(ordered))
        marker_width = max(16, CAIRO.text_width(self.context, marker) + 6)
        lines = self.wrap(clean_markdown(text), self.content_width - marker_width, 8.8)
        line_height = 11.7
        self.ensure(len(lines) * line_height + 3)
        CAIRO.text(self.context, self.MARGIN_X + 4, self.y + 8.8, marker, "#0f766e")
        for index, line in enumerate(lines):
            CAIRO.font(self.context, 8.8)
            CAIRO.text(self.context, self.MARGIN_X + marker_width, self.y + 8.8, line, "#1e293b")
            self.y += line_height
        self.y += 2

    def code(self, lines: list[str]) -> None:
        size = 8.0
        height = 18 + len(lines) * 11
        self.ensure(height + 8)
        CAIRO.rgb(self.context, "#0f172a")
        CAIRO.lib.cairo_rectangle(self.context, self.MARGIN_X, self.y, self.content_width, height)
        CAIRO.lib.cairo_fill(self.context)
        CAIRO.lib.cairo_select_font_face(self.context, b"DejaVu Sans Mono", 0, 0)
        CAIRO.lib.cairo_set_font_size(self.context, size)
        for index, line in enumerate(lines):
            CAIRO.text(self.context, self.MARGIN_X + 12, self.y + 18 + index * 11, line, "#e2e8f0")
        self.y += height + 8

    def image(self, path: Path, caption: str | None = None) -> None:
        if path.suffix.lower() != ".png" or not path.exists():
            return
        image_surface = CAIRO.lib.cairo_image_surface_create_from_png(str(path).encode())
        image_width = CAIRO.lib.cairo_image_surface_get_width(image_surface)
        image_height = CAIRO.lib.cairo_image_surface_get_height(image_surface)
        full_page_height = self.PAGE_HEIGHT - self.TOP - self.BOTTOM - 30
        remaining_height = self.PAGE_HEIGHT - self.BOTTOM - self.y - 24
        # Reuse a meaningful amount of remaining space. Previously every image
        # after introductory text was forced onto a fresh page, producing large
        # blank regions. If the remainder is too small for a legible figure,
        # start a new page; otherwise scale the figure into the current page.
        if remaining_height < 185:
            self.new_page()
            remaining_height = full_page_height
        max_height = min(full_page_height, remaining_height)
        scale = min(self.content_width / image_width, max_height / image_height)
        draw_width = image_width * scale
        draw_height = image_height * scale
        x = (self.PAGE_WIDTH - draw_width) / 2
        CAIRO.lib.cairo_save(self.context)
        CAIRO.lib.cairo_translate(self.context, x, self.y)
        CAIRO.lib.cairo_scale(self.context, scale, scale)
        CAIRO.lib.cairo_set_source_surface(self.context, image_surface, 0, 0)
        CAIRO.lib.cairo_paint(self.context)
        CAIRO.lib.cairo_restore(self.context)
        self.y += draw_height + 7
        CAIRO.lib.cairo_surface_destroy(image_surface)
        if caption:
            self.paragraph(caption, size=7.8, color="#64748b", italic=True, before=0, after=5)

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        column_count = max(len(row) for row in rows)
        rows = [row + [""] * (column_count - len(row)) for row in rows]
        character_weights = [max(5, max(len(clean_markdown(row[col])) for row in rows)) for col in range(column_count)]
        total_weight = sum(character_weights)
        widths = [self.content_width * weight / total_weight for weight in character_weights]
        minimum = 42 if column_count <= 6 else 29
        widths = [max(minimum, width) for width in widths]
        width_sum = sum(widths)
        widths = [width * self.content_width / width_sum for width in widths]
        font_size = 7.8 if column_count <= 5 else 6.8 if column_count <= 8 else 5.8

        def row_layout(row: list[str]) -> tuple[list[list[str]], float]:
            wrapped = [self.wrap(clean_markdown(cell), max(10, widths[index] - 7), font_size, bold=False) for index, cell in enumerate(row)]
            height = max(len(lines) for lines in wrapped) * font_size * 1.35 + 8
            return wrapped, height

        for row_index, row in enumerate(rows):
            wrapped, height = row_layout(row)
            if self.y + height > self.PAGE_HEIGHT - self.BOTTOM:
                self.new_page()
                if row_index > 1:
                    header_wrapped, header_height = row_layout(rows[0])
                    self._table_row(header_wrapped, widths, header_height, font_size, True)
            self._table_row(wrapped, widths, height, font_size, row_index == 0)
        self.y += 7

    def _table_row(self, wrapped: list[list[str]], widths: list[float], height: float, font_size: float, header: bool) -> None:
        x = self.MARGIN_X
        for index, lines in enumerate(wrapped):
            CAIRO.rgb(self.context, "#0f766e" if header else "#f8fafc")
            CAIRO.lib.cairo_rectangle(self.context, x, self.y, widths[index], height)
            CAIRO.lib.cairo_fill_preserve(self.context)
            CAIRO.rgb(self.context, "#cbd5e1")
            CAIRO.lib.cairo_set_line_width(self.context, 0.6)
            CAIRO.lib.cairo_stroke(self.context)
            CAIRO.font(self.context, font_size, bold=header)
            for line_index, line in enumerate(lines):
                CAIRO.text(
                    self.context,
                    x + 4,
                    self.y + 5 + font_size + line_index * font_size * 1.35,
                    line,
                    "#ffffff" if header else "#1e293b",
                )
            x += widths[index]
        self.y += height

    def finish(self) -> None:
        self._footer()
        CAIRO.lib.cairo_show_page(self.context)
        CAIRO.lib.cairo_surface_finish(self.surface)
        CAIRO.lib.cairo_destroy(self.context)
        CAIRO.lib.cairo_surface_destroy(self.surface)


def render_markdown_pdf(markdown_path: Path, output_path: Path) -> None:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    renderer = PdfRenderer(output_path)
    index = 0
    pending_caption: str | None = None
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        image_match = re.search(r'<img\s+src="([^"]+)"', stripped)
        if image_match:
            image_path = (markdown_path.parent / image_match.group(1)).resolve()
            caption = None
            block_end = index
            for lookahead in range(index + 1, min(index + 8, len(lines))):
                if "<em>" in lines[lookahead]:
                    caption = clean_markdown(lines[lookahead])
                block_end = lookahead
                if "</p>" in lines[lookahead]:
                    break
            renderer.image(image_path, caption)
            index = block_end + 1
            continue
        if stripped.startswith("<"):
            index += 1
            continue
        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            renderer.code(code_lines)
            index += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            renderer.heading(len(heading.group(1)), heading.group(2))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in rows[1]):
                rows.pop(1)
            renderer.table(rows)
            continue
        blockquote = re.match(r"^>\s?(.*)$", stripped)
        if blockquote:
            renderer.paragraph(blockquote.group(1), size=9.4, color="#0f766e", italic=True, indent=14, before=4, after=8)
            index += 1
            continue
        bullet = re.match(r"^-\s+(.*)$", stripped)
        ordered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if bullet:
            renderer.bullet(bullet.group(1))
            index += 1
            continue
        if ordered:
            renderer.bullet(ordered.group(2), ordered.group(1))
            index += 1
            continue
        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith(("#", "|", "- ", ">", "```", "<")) or re.match(r"^\d+\.\s+", candidate):
                break
            paragraph_lines.append(candidate)
            index += 1
        renderer.paragraph(" ".join(paragraph_lines))
    renderer.finish()


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary = ASSET_DIR / "hive_v2_summary_23042026.png"
    diagnostics = ASSET_DIR / "hive_v2_diagnostics_23042026.png"
    draw_summary_png(summary)
    draw_diagnostics_png(diagnostics)
    render_markdown_pdf(REPORT_MD, REPORT_PDF)
    print(f"Summary:     {summary}")
    print(f"Diagnostics: {diagnostics}")
    print(f"PDF:         {REPORT_PDF}")


if __name__ == "__main__":
    main()
