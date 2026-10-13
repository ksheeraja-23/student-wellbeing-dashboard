"""Export every report figure as interactive HTML and static PNG into outputs/figures/.
PNG export needs Chrome/Chromium for kaleido (set BROWSER_PATH=/path/to/chrome if it is not found)."""
from pathlib import Path
from src.preprocess import load_clean
from src.charts import all_report_figures

out = Path(__file__).parent / "outputs" / "figures"
out.mkdir(parents=True, exist_ok=True)
png_ok = True
for name, fig in all_report_figures(load_clean()).items():
    fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")
    if png_ok:
        try:
            fig.write_image(out / f"{name}.png", width=1100, height=fig.layout.height or 450, scale=2)
        except Exception as e:
            png_ok = False
            print("PNG export skipped (needs Chrome for kaleido):", str(e).splitlines()[0][:100])
print("Figures written to", out)
