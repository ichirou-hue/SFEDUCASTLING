"""Keep GPU, network and pretrained-model checks opt-in during normal CI."""
from pathlib import Path
import sys


def pytest_addoption(parser):
    parser.addoption("--run-ml-integration", action="store_true", default=False,
                     help="Collect ML integration checks requiring model weights or external services")


def pytest_ignore_collect(collection_path, config):
    integration = Path(__file__).resolve().parent / "integration"
    return integration in collection_path.parents and not config.getoption("--run-ml-integration")


# Shared imports for the relocated ChessCLIP integration suite.
sys.path.insert(0, str(Path(__file__).resolve().parent / "integration/chessclip"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "ml/synthesis_pipeline/rag_module_chessclip/chessclip/src"))
