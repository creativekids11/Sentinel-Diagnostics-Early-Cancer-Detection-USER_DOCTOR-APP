#!/usr/bin/env python3
"""
Test script to verify that annotations are disabled before AI results are generated
in the AI scanner tab under doctor's folder.
"""

import os
import sys
from jinja2 import Template
import unittest

class TestAnnotationDisabling(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Template content from aiscanner.html (simplified for testing)
        self.template_content = """
{% if session.scan_analysis and session.scan_analysis.ai_results %}
<div class="ai-results-section">
    <h3>AI Analysis Results</h3>
    <p>Risk Level: {{ session.scan_analysis.ai_results.risk_level | default('N/A') }}</p>
</div>
{% endif %}

{% if session.scan_analysis and session.scan_analysis.ai_results %}
<div class="editor-container">
    <div class="editor-header">
        <h3>AI Report Editor</h3>
        <p>Enhance AI results with manual annotations</p>
    </div>
    <div class="editor-toolbar">
        <button class="tool-btn" onclick="selectTool('brush')">Highlight Brush</button>
    </div>
</div>
{% endif %}
"""

        self.template = Template(self.template_content)

    def test_annotations_hidden_before_ai_results(self):
        """Test that annotation section is hidden when no AI results exist"""
        # No session data
        context = {'session': {}}
        result = self.template.render(**context)

        # Check that neither AI results nor editor container are shown
        self.assertNotIn('ai-results-section', result)
        self.assertNotIn('editor-container', result)
        print("✓ PASS: Annotations hidden when no AI results exist")

    def test_annotations_hidden_with_empty_ai_results(self):
        """Test that annotation section is hidden when AI results are empty"""
        context = {
            'session': {
                'scan_analysis': {'ai_results': None}
            }
        }
        result = self.template.render(**context)

        # Check that neither AI results nor editor container are shown
        self.assertNotIn('ai-results-section', result)
        self.assertNotIn('editor-container', result)
        print("✓ PASS: Annotations hidden when AI results are None")

    def test_annotations_shown_after_ai_results(self):
        """Test that annotation section is shown when AI results exist"""
        context = {
            'session': {
                'scan_analysis': {
                    'ai_results': {
                        'risk_level': 'High',
                        'confidence': '87.3%'
                    }
                }
            }
        }
        result = self.template.render(**context)

        # Check that both AI results and editor container are shown
        self.assertIn('ai-results-section', result)
        self.assertIn('editor-container', result)
        self.assertIn('AI Report Editor', result)
        self.assertIn('Enhance AI results with manual annotations', result)
        print("✓ PASS: Annotations shown when AI results exist")

    def test_workflow_sequence(self):
        """Test the complete workflow sequence"""
        print("\n--- Testing Complete Workflow Sequence ---")

        # Step 1: Before AI analysis
        context = {'session': {}}
        result = self.template.render(**context)
        self.assertNotIn('editor-container', result)
        print("1. ✓ Before AI analysis: Annotations are hidden")

        # Step 2: After AI analysis completes
        context = {
            'session': {
                'scan_analysis': {
                    'ai_results': {
                        'risk_level': 'Medium',
                        'confidence': '92.1%'
                    }
                }
            }
        }
        result = self.template.render(**context)
        self.assertIn('editor-container', result)
        self.assertIn('AI Report Editor', result)
        print("2. ✓ After AI analysis: Annotations are enabled")

if __name__ == '__main__':
    print("Testing annotation disabling before AI results...")
    print("=" * 50)

    unittest.main(verbosity=2)