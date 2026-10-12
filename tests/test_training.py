"""Exercise training, logged metrics, checkpoint loading, and generation on CPU."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from generate import sample_tokens
from model import ModelConfig, ShakespeareTransformer
from train import estimate_loss


class TrainingTests(unittest.TestCase):
    def test_training_cli_checkpoint_and_generation(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'input.txt'
            data.write_text('To be or ' * 100, encoding='utf-8')
            command = [
                sys.executable, str(project / 'train.py'), '--data-path', str(data),
                '--checkpoint-dir', str(root / 'checkpoints'), '--device', 'cpu',
                '--max-steps', '40', '--eval-interval', '20', '--eval-iters', '3',
                '--batch-size', '4', '--block-size', '8', '--n-embd', '8',
                '--n-head', '2', '--n-layer', '1', '--dropout', '0',
                '--learning-rate', '0.02', '--sample-tokens', '10',
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
            self.assertIn('validation', result.stdout)
            checkpoint_path = next((root / 'checkpoints').glob('*/model.pt'))
            with (checkpoint_path.parent / 'losses.csv').open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([int(row['step']) for row in rows], [0, 20, 40])
            self.assertLess(float(rows[-1]['train_loss']), float(rows[0]['train_loss']))
            self.assertLess(float(rows[-1]['validation_loss']), float(rows[0]['validation_loss']))
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(checkpoint['step'], 40)
            model = ShakespeareTransformer(ModelConfig(**checkpoint['config']))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            tokens = torch.tensor([[checkpoint['stoi']['T']]])
            with torch.no_grad():
                expected, _ = model(tokens)
            second_model = ShakespeareTransformer(ModelConfig(**checkpoint['config']))
            second_model.load_state_dict(checkpoint['model_state_dict'])
            second_model.eval()
            with torch.no_grad():
                actual, _ = second_model(tokens)
            torch.testing.assert_close(actual, expected)
            generated = subprocess.run([
                sys.executable, str(project / 'generate.py'), '--checkpoint', str(checkpoint_path),
                '--device', 'cpu', '--prompt', 'To', '--max-new-tokens', '20',
            ], capture_output=True, text=True, check=True, timeout=60)
            self.assertTrue(generated.stdout.startswith('To'))
            self.assertEqual(len(generated.stdout.rstrip('\n')), 22)

    def test_sampling_crops_context_and_restores_mode(self) -> None:
        model = ShakespeareTransformer(ModelConfig(
            vocab_size=6, block_size=4, n_embd=8, n_head=2, n_layer=1,
        ))
        tokens = torch.tensor([[0, 1, 2, 3, 4, 5]])
        result = sample_tokens(model, tokens, 7, temperature=0.8)
        self.assertEqual(result.shape, (1, 13))
        torch.testing.assert_close(result[:, :6], tokens)
        self.assertTrue(model.training)
        for temperature in (0, -1, float('inf'), float('nan')):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                sample_tokens(model, tokens, 1, temperature)

    def test_validation_disables_gradients_and_restores_mode(self) -> None:
        model = ShakespeareTransformer(ModelConfig(
            vocab_size=6, block_size=4, n_embd=8, n_head=2, n_layer=1,
        ))
        splits = {'train': torch.arange(30) % 6, 'validation': torch.arange(20) % 6}
        losses = estimate_loss(model, splits, 2, 2, 'cpu')
        self.assertEqual(set(losses), set(splits))
        self.assertTrue(all(value > 0 for value in losses.values()))
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
