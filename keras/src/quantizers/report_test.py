from keras.src import testing
from keras.src.quantizers.report import QuantizationReport


class QuantizationReportTest(testing.TestCase):
    def test_empty_report(self):
        report = QuantizationReport(mode="int8")
        self.assertEqual(report.mode, "int8")
        self.assertEqual(report.num_quantized, 0)
        self.assertEqual(report.num_skipped, 0)
        self.assertEqual(report.num_errors, 0)
        self.assertIsNone(report.summary_warning())

    def test_add_and_query(self):
        report = QuantizationReport(mode="int8")
        report.add_quantized("dense", "int8", "int8_from_float32")
        report.add_skipped("act", QuantizationReport.SKIP_NO_SUPPORT)
        report.add_skipped("bn", QuantizationReport.SKIP_FILTERED)
        report.add_skipped("d2", QuantizationReport.SKIP_ALREADY_QUANTIZED)

        self.assertEqual(report.num_quantized, 1)
        self.assertEqual(report.num_skipped, 3)
        self.assertEqual(
            report.quantized, [("dense", "int8", "int8_from_float32")]
        )
        self.assertEqual(
            report.skipped_by_reason(QuantizationReport.SKIP_NO_SUPPORT),
            ["act"],
        )
        self.assertEqual(
            report.skipped_by_reason(QuantizationReport.SKIP_FILTERED), ["bn"]
        )
        self.assertEqual(
            report.skipped_by_reason(QuantizationReport.SKIP_ALREADY_QUANTIZED),
            ["d2"],
        )

    def test_summary_warning_reports_unsupported(self):
        report = QuantizationReport(mode="int8")
        report.add_quantized("dense", "int8", "int8_from_float32")
        report.add_skipped("act", QuantizationReport.SKIP_NO_SUPPORT)
        report.add_skipped("input", QuantizationReport.SKIP_NO_SUPPORT)
        # Filtered layers must not trigger a warning.
        report.add_skipped("bn", QuantizationReport.SKIP_FILTERED)

        message = report.summary_warning()
        self.assertIsNotNone(message)
        self.assertIn("2 layer(s) were skipped", message)
        self.assertIn("'act'", message)
        self.assertIn("'input'", message)
        # Filtered layers are not mentioned as unsupported.
        self.assertNotIn("'bn'", message)

    def test_summary_warning_none_when_only_filtered(self):
        report = QuantizationReport(mode="int8")
        report.add_quantized("dense", "int8", "int8_from_float32")
        report.add_skipped("bn", QuantizationReport.SKIP_FILTERED)
        report.add_skipped("d2", QuantizationReport.SKIP_ALREADY_QUANTIZED)
        self.assertIsNone(report.summary_warning())

    def test_summary_warning_caps_examples(self):
        report = QuantizationReport(mode="int8")
        for i in range(8):
            report.add_skipped(f"layer_{i}", QuantizationReport.SKIP_NO_SUPPORT)
        message = report.summary_warning(max_examples=5)
        self.assertIn("8 layer(s) were skipped", message)
        self.assertIn("...", message)
        # Only the first five names appear.
        self.assertIn("'layer_0'", message)
        self.assertIn("'layer_4'", message)
        self.assertNotIn("'layer_5'", message)

    def test_render_contains_sections(self):
        report = QuantizationReport(mode="int8")
        report.add_quantized("dense", "int8", "int8_from_float32")
        report.add_skipped("act", QuantizationReport.SKIP_NO_SUPPORT)
        rendered = report.render()
        self.assertIn("Quantization report (mode='int8')", rendered)
        self.assertIn("Quantized 1 layer(s):", rendered)
        self.assertIn("dense (int8): int8_from_float32", rendered)
        self.assertIn("Skipped 1 layer(s):", rendered)
        self.assertIn("act: no quantize support", rendered)

    def test_repr(self):
        report = QuantizationReport(mode="int4")
        report.add_quantized("dense", "int4", "int4/-1_from_float32")
        self.assertIn("mode='int4'", repr(report))
        self.assertIn("quantized=1", repr(report))

    def test_skip_reasons_exposed_as_class_attributes(self):
        # The skip reasons are available as class attributes on
        # `QuantizationReport`.
        self.assertEqual(
            QuantizationReport.SKIP_NO_SUPPORT, "no quantize support"
        )
        self.assertEqual(QuantizationReport.SKIP_FILTERED, "filtered out")
        self.assertEqual(
            QuantizationReport.SKIP_ALREADY_QUANTIZED, "already quantized"
        )
        self.assertEqual(
            QuantizationReport.SKIP_OUTSIDE_STRUCTURE,
            "outside quantization structure",
        )
