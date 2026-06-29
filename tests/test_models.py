import unittest

from decopatch.models.factory import build_model
from decopatch.models.registry import available_models


class ModelTest(unittest.TestCase):
    def test_registry_has_defaults(self):
        names = available_models()
        self.assertIn("decopatch_tiny", names)
        self.assertIn("decopatch_base", names)

    def test_regressor_predicts_scalar(self):
        model = build_model("decopatch_tiny")
        prediction = model.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertIsInstance(prediction, float)

    def test_classifier_predicts_distribution(self):
        model = build_model("decopatch_cls")
        prediction = model.predict([0.1 for _ in range(16)])
        self.assertEqual(len(prediction), 2)
        self.assertAlmostEqual(sum(prediction), 1.0)

    def test_reconstructor_predicts_vector(self):
        model = build_model("decopatch_rec32")
        prediction = model.predict([0.1 for _ in range(32)])
        self.assertEqual(len(prediction), 32)


if __name__ == "__main__":
    unittest.main()
