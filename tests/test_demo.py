"""Ensure the portable public export reproduces the frozen experiment."""
import json
import unittest

from src.demo import ARTIFACTS, evaluate


class DemoTests(unittest.TestCase):
    def test_public_snapshot_reproduces_report(self):
        actual = evaluate()
        report = json.loads((ARTIFACTS / 'results.json').read_text())
        self.assertEqual(actual['examples'], report['n'])
        self.assertEqual(actual['L'], report['L'])
        self.assertAlmostEqual(actual['majority_class_accuracy'], .735)
        for metric in ['accuracy', 'balanced_accuracy', 'roc_auc']:
            self.assertAlmostEqual(actual[metric], report['metrics']['cnn'][metric], places=6)
        self.assertAlmostEqual(actual['curve_mae'], report['curve_mae']['cnn'], places=6)
