import os
import sys
import json
import torch

# Подключаем локальный chessclip/src
current_dir = os.path.dirname(os.path.abspath(__file__))
chessclip_src = os.path.join(current_dir, "chessclip", "src")
if chessclip_src in sys.path:
    sys.path.remove(chessclip_src)
sys.path.insert(0, chessclip_src)

from open_clip.model import ChessCLIP, CLIPVisionCfg, CLIPTextCfg
from board_converter import fen_to_lc0_112_planes

DEFAULT_WEIGHTS_PATH = os.path.expanduser("~/SFEDUCASTLING/ml/models/chessclip/chessclip.pt")


class ChessPositionEncoder:
    def __init__(self, weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        weights_path = os.path.abspath(os.path.expanduser(weights_path))
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")

        config_path = os.path.join(chessclip_src, "open_clip", "model_configs", "chessclip-quickgelu.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        chess_vision_cfg = CLIPVisionCfg(**cfg["chess_vision_cfg"])
        text_cfg = CLIPTextCfg(**cfg["text_cfg"])

        self.model = ChessCLIP(
            embed_dim=cfg["embed_dim"],
            chess_vision_cfg=chess_vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=cfg.get("quick_gelu", True)
        ).to(self.device)

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        print("[✓] ChessCLIP готов к получению эмбеддингов!")

    @torch.no_grad()
    def get_embedding(self, fen: str) -> list[float]:
        tensor = fen_to_lc0_112_planes(fen).to(self.device)  # (1, 112, 8, 8)
        dummy_action = torch.zeros((tensor.shape[0], 1858), dtype=torch.float32, device=self.device)
        
        features = self.model.visual(tensor, dummy_action)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()