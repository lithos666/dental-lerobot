"""Generate a 9x6 chessboard calibration pattern as a PDF.

Prints at a physical size where each square is exactly SQUARE_SIZE_M meters
(25 mm by default), so the printed board matches the value hard-coded in
calibrate_scene_camera.py regardless of printer scaling.

Usage:
    python dental_robot/generate_chessboard.py

The output PDF includes:
  - Print instructions on page 1 (verify with ruler, glue to rigid backing).
  - The chessboard itself on page 2, sized to 25 mm squares at any DPI
    because we emit PostScript points (1 pt = 1/72 inch = 0.3528 mm).

Why PDF not PNG: PDF stores dimensions in absolute PostScript points, so
even if the print driver applies "fit to page" the chessboard is sized
correctly *only if* you choose A4 and "actual size / 100%" in the dialog.
If your printer insists on scaling, prefer PNG + "actual size" instead.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas

# Must match calibrate_scene_camera.py exactly.
CHESSBOARD = (9, 6)         # inner corners
SQUARE_SIZE_M = 0.025       # 25 mm; each black/white square = 25 mm
MARGIN_MM = 15              # white border around the pattern

OUT_PDF = Path(__file__).parent / "chessboard_9x6.pdf"


def draw_chessboard(c: pdf_canvas.Canvas, x0: float, y0: float, square_mm: float,
                    cols: int, rows: int):
    """Draw a cols x rows chessboard with bottom-left corner at (x0, y0)."""
    for r in range(rows):
        for col in range(cols):
            if (r + col) % 2 == 0:
                continue  # white square -> skip
            x = x0 + col * square_mm
            y = y0 + (rows - 1 - r) * square_mm
            c.rect(x, y, square_mm, square_mm, stroke=0, fill=1)


def main():
    """CLI entry point: render the calibration chessboard to a printable PDF."""
    cols_inner, rows_inner = CHESSBOARD
    cols_squares = cols_inner + 1
    rows_squares = rows_inner + 1
    square = SQUARE_SIZE_M * 1000.0 * mm  # convert meters -> PostScript points via mm

    page_w, page_h = A4

    # --- Page 1: instructions ---
    c = pdf_canvas.Canvas(str(OUT_PDF), pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, page_h - 30 * mm, "Chessboard calibration pattern (9 x 6 inner corners)")
    c.setFont("Helvetica", 11)
    lines = [
        f"Inner corners: {cols_inner} cols x {rows_inner} rows = {cols_inner * rows_inner} corners",
        f"Square size:   {SQUARE_SIZE_M * 1000:.1f} mm x {SQUARE_SIZE_M * 1000:.1f} mm (BLACK squares)",
        f"Total squares: {cols_squares} x {rows_squares} = {cols_squares * rows_squares} (10x7)",
        "",
        "IMPORTANT:",
        " 1. Print at 100% / Actual Size (do NOT choose 'Fit to page').",
        " 2. Measure one black square with a ruler: it MUST be 25 mm.",
        "    If not, your driver scaled the print; re-print with 'Actual size'.",
        " 3. Glue the paper to a rigid flat surface (cardboard / foam board).",
        " 4. Do not trim or distort the printed area.",
    ]
    y = page_h - 50 * mm
    for line in lines:
        c.drawString(20 * mm, y, line)
        y -= 6 * mm

    c.showPage()

    # --- Page 2: chessboard ---
    board_w = cols_squares * square
    board_h = rows_squares * square
    # Centre on the page with margin around.
    x0 = (page_w - board_w) / 2
    y0 = (page_h - board_h) / 2

    # Optional: print a small ruler tick so you can sanity-check the scale.
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    draw_chessboard(c, x0, y0, square, cols_squares, rows_squares)

    # Print a 100 mm scale bar on the side for visual verification.
    bar_x = x0 - 10 * mm
    bar_y = y0
    c.line(bar_x, bar_y, bar_x, bar_y + 100 * mm)
    for tick in range(0, 101, 10):
        ty = bar_y + tick * mm
        c.line(bar_x - 1 * mm, ty, bar_x + 1 * mm, ty)
    c.setFont("Helvetica", 7)
    c.drawString(bar_x - 6 * mm, bar_y, "0")
    c.drawString(bar_x - 9 * mm, bar_y + 100 * mm, "100mm")

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0, y0 - 8 * mm, "Each black square = 25 mm (verify with ruler!)")
    c.showPage()
    c.save()

    print(f"Saved {OUT_PDF}")
    print(f"  Inner corners: {cols_inner}x{rows_inner}")
    print(f"  Square size:   {SQUARE_SIZE_M * 1000:.1f} mm")
    print(f"  Board size:    {board_w / mm:.1f} mm x {board_h / mm:.1f} mm")
    print()
    print("PRINT INSTRUCTIONS:")
    print("  1. Open chessboard_9x6.pdf in any PDF viewer.")
    print("  2. Print with 'Actual size' / '100%' (NEVER 'Fit to page').")
    print("  3. Use the printed ruler tick to verify scale.")


if __name__ == "__main__":
    main()
