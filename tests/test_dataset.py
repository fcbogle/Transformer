"""Small checks for raw corpus loading and character tokenisation."""

import tempfile
import unittest
from pathlib import Path

from dataset import CharacterTokenizer, load_raw_text


class CharacterTokenizerTests(unittest.TestCase):
    def test_round_trip_preserves_unicode_and_whitespace(self) -> None:
        text = "ROMEO:\n  O, café!\r\n"
        tokenizer = CharacterTokenizer(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.vocab_size, len(set(text)))
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def test_mapping_is_sorted_and_independent_of_character_order(self) -> None:
        tokenizer = CharacterTokenizer("aba ")
        self.assertEqual(tokenizer.stoi, {" ": 0, "a": 1, "b": 2})
        self.assertEqual(tokenizer.itos, {0: " ", 1: "a", 2: "b"})
        self.assertEqual(tokenizer.stoi, CharacterTokenizer(" ba").stoi)

    def test_empty_vocabulary_and_unknown_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CharacterTokenizer("")
        tokenizer = CharacterTokenizer("abc")
        with self.assertRaises(KeyError):
            tokenizer.encode("z")
        with self.assertRaises(KeyError):
            tokenizer.decode([3])


class RawTextTests(unittest.TestCase):
    def test_load_preserves_raw_text(self) -> None:
        text = "O, café!\r\n  ROMEO:\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_bytes(text.encode("utf-8"))
            self.assertEqual(load_raw_text(path), text)

    def test_missing_and_empty_corpora_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            with self.assertRaisesRegex(FileNotFoundError, "download instructions"):
                load_raw_text(path)
            path.touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                load_raw_text(path)


if __name__ == "__main__":
    unittest.main()
