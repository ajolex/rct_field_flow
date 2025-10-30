from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
import weasyprint


@dataclass
class ReportConfig:
    template_path: str
    output_dir: str = "reports"
    output_filename: str = "weekly_report"
    render_pdf: bool = True
    render_html: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict | None) -> "ReportConfig":
        if not cfg:
            return cls(template_path="templates/weekly_report.html")
        return cls(
            template_path=cfg.get("weekly_template", "templates/weekly_report.html"),
            output_dir=cfg.get("output_dir", "reports"),
            output_filename=cfg.get("output_filename", "weekly_report"),
            render_pdf=cfg.get("render_pdf", True),
            render_html=cfg.get("render_html", True),
        )


def generate_weekly_report(context: Dict, config: Dict | None = None) -> Dict[str, Path]:
    """Render the weekly monitoring report to HTML/PDF."""
    cfg = ReportConfig.from_dict(config)
    template_path = Path(cfg.template_path)
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_path.name)
    html = template.render(**context)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}
    if cfg.render_html:
        html_path = output_dir / f"{cfg.output_filename}.html"
        html_path.write_text(html, encoding="utf-8")
        outputs["html"] = html_path

    if cfg.render_pdf:
        pdf_path = output_dir / f"{cfg.output_filename}.pdf"
        weasyprint.HTML(string=html).write_pdf(str(pdf_path))
        outputs["pdf"] = pdf_path

    return outputs
