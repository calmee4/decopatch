import unittest

from decopatch.core import chunk, decompose_patch, multiscale


class CoreTest(unittest.TestCase):
    def test_chunk_reconstructs_covered_signal(self):
        patches = chunk([1, 2, 3, 4], 2, 2)
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches[0].values, (1.0, 2.0))

    def test_decomposition_compose(self):
        patch = chunk([1, 2, 3], 3)[0]
        decomposition = decompose_patch(patch)
        self.assertEqual(len(decomposition.compose()), 3)

    def test_multiscale_levels(self):
        grids = multiscale([1, 2, 3, 4, 5, 6, 7, 8], (2, 4))
        self.assertEqual(len(grids), 2)
        self.assertGreater(grids[0].coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
