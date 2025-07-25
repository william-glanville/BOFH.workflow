import os
import markdown

class ReportRenderer:
    def __init__(self, template_path: str):
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

    def render(self, md_text: str) -> str:
        html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        return self.template.replace("{html_body}", html_body)

