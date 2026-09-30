"""Verify tensor conversion, split boundaries, and next-character targets."""

import unittest

import torch

from dataset import (
    DEFAULT_DATA_PATH,
    CharacterTokenizer,
    encode_text,
    get_batch,
    load_raw_text,
    split_data,
)


class TensorPreparationTests(unittest.TestCase):
    def test_encoding_preserves_text_as_integer_tensor(self) -> None:
        text = "ROMEO:\nTo be, or not to be."
        tokenizer = CharacterTokenizer(text)
        tokens = encode_text(text, tokenizer)
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens.device.type, "cpu")
        self.assertEqual(tokens.shape, (len(text),))
        self.assertEqual(tokenizer.decode(tokens.tolist()), text)

    @unittest.skipUnless(DEFAULT_DATA_PATH.is_file(), "Download data/input.txt first")
    def test_full_dataset_tensor(self) -> None:
        """Verify the full dataset tensor is one-dimensional and on CPU."""
        text = load_raw_text()
        tokenizer = CharacterTokenizer(text)
        tokens = encode_text(text, tokenizer)
        self.assertEqual(tokens.ndim, 1)
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens.device.type, "cpu")
        self.assertEqual(len(tokens), len(text))

    def test_split_preserves_order_and_rounds_training_length_down(self) -> None:
        tokens = torch.arange(101)
        train, validation = split_data(tokens)
        self.assertEqual(len(train), 90)
        self.assertEqual(len(validation), 11)
        self.assertEqual(train.tolist(), list(range(90)))
        self.assertEqual(validation.tolist(), list(range(90, 101)))

    def test_split_rejects_insufficient_or_malformed_data(self) -> None:
        for tokens in (torch.tensor([], dtype=torch.long), torch.tensor([0]),
                       torch.zeros(2, 3, dtype=torch.long), torch.zeros(10)):
            with self.subTest(shape=tokens.shape, dtype=tokens.dtype):
                with self.assertRaises(ValueError):
                    split_data(tokens)


class BatchTests(unittest.TestCase):
    def setUp(self) -> None:
        # Restore the RNG afterwards so these tests do not affect other tests.
        self.random_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, self.random_state)
        torch.manual_seed(1337)

    def test_batches_have_expected_shape_and_shift_without_crossing_splits(self) -> None:
        train, validation = split_data(torch.arange(1000))
        for split in (train, validation):
            with self.subTest(first_token=split[0].item()):
                x, y = get_batch(split, batch_size=32, block_size=16)
                self.assertEqual(x.shape, (32, 16))
                self.assertEqual(y.shape, (32, 16))
                self.assertEqual(x.dtype, torch.long)
                self.assertEqual(y.dtype, torch.long)
                self.assertEqual(x.device.type, "cpu")
                self.assertEqual(y.device.type, "cpu")
                # Sequential IDs let us check every target, including the last.
                self.assertTrue(torch.equal(y, x + 1))
                self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
                self.assertGreaterEqual(x.min().item(), split[0].item())
                self.assertLessEqual(y.max().item(), split[-1].item())

    def test_minimum_length_split_includes_last_target(self) -> None:
        tokenizer = CharacterTokenizer("ROMEO:")
        tokens = encode_text("ROMEO:", tokenizer)
        x, y = get_batch(tokens, batch_size=2, block_size=5)
        self.assertEqual(tokenizer.decode(x[0].tolist()), "ROMEO")
        self.assertEqual(tokenizer.decode(y[0].tolist()), "OMEO:")
        self.assertTrue(torch.equal(x[0], x[1]))

    def test_seed_reproduces_batches(self) -> None:
        tokens = torch.arange(100)
        first = get_batch(tokens, 4, 8)
        torch.manual_seed(1337)
        second = get_batch(tokens, 4, 8)
        for before, after in zip(first, second):
            self.assertTrue(torch.equal(before, after))

    def test_invalid_batch_sizes_and_insufficient_context_are_rejected(self) -> None:
        for batch_size, block_size in ((0, 4), (2, 0), (-1, 4), (2, -1), (2, 10)):
            with self.subTest(batch_size=batch_size, block_size=block_size):
                with self.assertRaises(ValueError):
                    get_batch(torch.arange(10), batch_size, block_size)

    def test_malformed_token_tensors_are_rejected(self) -> None:
        for tokens in (torch.zeros(10), torch.zeros(2, 10, dtype=torch.long)):
            with self.subTest(shape=tokens.shape, dtype=tokens.dtype):
                with self.assertRaises(ValueError):
                    get_batch(tokens, 2, 4)


if __name__ == "__main__":
    unittest.main()
