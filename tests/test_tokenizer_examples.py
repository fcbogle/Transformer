"""Printed examples for exploring character tokenisation.

Run from the project root:
    python3 -m unittest discover -s tests -p 'test_tokenizer_examples.py' -v
"""

import unittest

from dataset import DEFAULT_DATA_PATH, CharacterTokenizer, load_raw_text


def print_example(text: str, tokenizer: CharacterTokenizer) -> None:
    """Use repr so spaces, quotation marks, and line breaks are visible."""
    tokens = tokenizer.encode(text)
    print(f"\nText (Python representation): {text!r}")
    print("The outer quotes mark the string; they are not extra characters.")
    print(f"Token IDs: {tokens}")
    print(f"Characters: {len(text)}; integers: {len(tokens)}")
    print("Character → integer:")
    for character, token in zip(text, tokens):
        print(f"  {character!r:>6} → {token}")
    print(f"Decoded text: {tokenizer.decode(tokens)!r}")


class ShakespeareExamplesTests(unittest.TestCase):
    """Inspect real data using the vocabulary from the entire corpus."""

    @unittest.skipUnless(DEFAULT_DATA_PATH.is_file(), "Download data/input.txt first")
    def test_first_line(self) -> None:
        corpus = load_raw_text()
        tokenizer = CharacterTokenizer(corpus)
        # Keep the line ending: it is a character with its own token ID too.
        first_line = corpus.splitlines(keepends=True)[0]

        print(f"\nCorpus vocabulary: {tokenizer.vocab_size} unique characters")
        print_example(first_line, tokenizer)

        tokens = tokenizer.encode(first_line)
        self.assertEqual(len(tokens), len(first_line))
        self.assertEqual(tokenizer.decode(tokens), first_line)


class TokenizerExperimentTests(unittest.TestCase):
    """Edit the example text here and rerun to explore the mappings."""

    def test_custom_text(self) -> None:
        # Change this string to experiment. The double quotes and newline below
        # are actual text; the surrounding single quotes are Python syntax.
        text = '"Hi Frank!"\n'
        # This vocabulary comes only from this example, so its IDs can differ
        # from the IDs assigned using the full Shakespeare corpus.
        tokenizer = CharacterTokenizer(text)

        print("\nExperiment: vocabulary built from the example text only")
        print_example(text, tokenizer)

        tokens = tokenizer.encode(text)
        self.assertEqual(len(tokens), len(text))
        self.assertEqual(tokenizer.decode(tokens), text)


if __name__ == "__main__":
    unittest.main()
