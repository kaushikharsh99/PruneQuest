from unittest.mock import patch, MagicMock
from prunequest.stage1.export.validator import CheckpointValidator


def test_validator_missing_directory():
    ok, msg = CheckpointValidator.validate("/nonexistent/directory/path")
    assert ok is False
    assert "does not exist" in msg


def test_validator_subprocess_mock():
    with patch("subprocess.run") as mock_run:
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "VALIDATION_OK|1500000|The sum is 30"
        mock_run.return_value = mock_res

        with patch("os.path.exists", return_value=True):
            ok, msg = CheckpointValidator.validate("/fake/checkpoint")
            assert ok is True
            assert "VALIDATION_OK" in msg
