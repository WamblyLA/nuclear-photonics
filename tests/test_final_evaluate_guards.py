import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.final_evaluate import main


class FinalEvaluationGuardTests(unittest.TestCase):
    def run_rejected_candidate(self, saved, embedded=None):
        # All paths are temporary; no checkpoint or official image file is read.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "candidate"
            run.mkdir()
            (run / "config.json").write_text(json.dumps(saved), encoding="utf-8")
            output = root / "final_test"
            arguments = ["final_evaluate", "--run", str(run), "--output-root", str(output), "--device", "cpu"]
            with patch("sys.argv", arguments), \
                 patch("src.final_evaluate.FinalTestBank", side_effect=AssertionError("Official test must stay closed")) as bank, \
                 patch("src.final_evaluate.build_model", side_effect=AssertionError("No model should be constructed")) as model, \
                 patch("src.final_evaluate.torch.load", return_value={"config": embedded}) as load:
                with self.assertRaisesRegex(ValueError, "diagnostic-only weights"):
                    main()
                bank.assert_not_called()
                model.assert_not_called()
                self.assertFalse(output.exists())
                return load.call_count

    def test_saved_diagnostic_flags_reject_before_checkpoint_loading(self):
        for flags in ({"diagnostic_only": True}, {"fixed_train_quartets": 32}):
            with self.subTest(flags=flags):
                self.assertEqual(self.run_rejected_candidate(flags), 0)

    def test_checkpoint_flags_cannot_be_hidden_by_clean_run_config(self):
        saved = {"diagnostic_only": False, "fixed_train_quartets": 0}
        for flags in ({"diagnostic_only": True}, {"fixed_train_quartets": 32}):
            with self.subTest(flags=flags):
                self.assertEqual(self.run_rejected_candidate(saved, embedded=flags), 1)


if __name__ == "__main__":
    unittest.main()
