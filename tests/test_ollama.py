import shutil
import subprocess
import unittest


class TestOllamaInstalled(unittest.TestCase):
    """Verify that the Ollama CLI is installed and responsive."""

    def test_ollama_on_path(self):
        """ollama executable should be found on PATH."""
        self.assertIsNotNone(
            shutil.which("ollama"),
            "ollama executable not found on PATH. "
            "Install it from https://ollama.com and make sure it is on your PATH.",
        )

    def test_ollama_version(self):
        """ollama --version should return successfully."""
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"'ollama --version' exited with code {result.returncode}. "
            f"stderr: {result.stderr.strip()}",
        )
        self.assertTrue(
            result.stdout.strip() or result.stderr.strip(),
            "ollama --version produced no output",
        )


if __name__ == "__main__":
    unittest.main()
