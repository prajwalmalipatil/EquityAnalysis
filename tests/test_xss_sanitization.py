"""
test_xss_sanitization.py
Tests both Python and Javascript sanitization mechanisms to prevent XSS.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

from src.services.macro_intelligence.enrichment_service import EnrichmentService


class TestXSSSanitization(unittest.TestCase):
    def test_python_prompt_sanitization(self):
        """Verify that EnrichmentService._sanitize_prompt_input strips HTML and template tags."""
        payload = 'Hello <script>alert(1)</script> """prompt injection"""'
        sanitized = EnrichmentService._sanitize_prompt_input(payload)
        self.assertEqual(sanitized, "Hello alert(1) prompt injection")

    def test_javascript_html_sanitization_via_node(self):
        """Runs the actual sanitizeHTML function from dashboard/app.js using node."""
        app_js_path = Path("dashboard/app.js").resolve()
        if not app_js_path.exists():
            self.skipTest("dashboard/app.js not found")

        # Create node evaluation script
        node_script = f"""
        const fs = require('fs');
        const code = fs.readFileSync('{app_js_path.as_posix()}', 'utf8');

        // Mock document element behavior for Node context
        const document = {{
            createElement: () => ({{
                set textContent(val) {{
                    this._val = val;
                }},
                get innerHTML() {{
                    return this._val
                        .replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                        .replace(/"/g, '&quot;')
                        .replace(/'/g, '&#039;');
                }}
            }})
        }};

        // Extract sanitizeHTML function from app.js to avoid loading DOMContentLoaded event listeners
        const match = code.match(/function\\s+sanitizeHTML\\s*\\([^\\)]*\\)\\s*\\{{[^\\}}]*\\}}/);
        if (!match) {{
            console.error("sanitizeHTML function not found in app.js");
            process.exit(1);
        }}
        
        eval(match[0]);

        const input = '<img src=x onerror=alert(1)>';
        const result = sanitizeHTML(input);
        console.log(result);
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(node_script)
            temp_name = f.name

        try:
            res = subprocess.run(["node", temp_name], capture_output=True, text=True, check=True)
            output = res.stdout.strip()
            # Assert '<' and '>' are escaped
            self.assertNotIn("<", output)
            self.assertNotIn(">", output)
            self.assertEqual(output, "&lt;img src=x onerror=alert(1)&gt;")
        finally:
            Path(temp_name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
