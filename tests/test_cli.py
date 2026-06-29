import unittest

from decopatch.cli import main


class CliTest(unittest.TestCase):
    def test_models_command(self):
        self.assertEqual(main(["models"]), 0)

    def test_run_command(self):
        self.assertEqual(main(["run", "--model", "decopatch_tiny"]), 0)


if __name__ == "__main__":
    unittest.main()
