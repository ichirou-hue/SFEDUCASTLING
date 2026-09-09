import os
from typing import List, Dict, Any, Optional
import chromadb
from clip_encoder import ChessPositionEncoder

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")


class ChessVectorStore:
    def __init__(
        self,
        persist_directory: str = DEFAULT_DB_DIR,
        collection_name: str = "chess_positions",
        encoder: Optional[ChessPositionEncoder] = None
    ):
        """
        Интерфейс локального векторного хранилища ChromaDB.
        Метрика расстояния: cosine (косинусное сходство).
        """
        self.persist_directory = os.path.abspath(os.path.expanduser(persist_directory))
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.encoder = encoder or ChessPositionEncoder()
        print(f"[✓] ChromaDB подключена. Коллекция: '{collection_name}' (Записей: {self.collection.count()})")

    def add_position(
        self,
        fen: str,
        move: str = "",
        eval_score: float = 0.0,
        game_id: str = "unknown",
        move_number: int = 1,
        comment: str = ""
    ):
        """Добавление единичной позиции с метаданными в индекс."""
        embedding = self.encoder.get_embedding(fen)
        pos_id = f"{game_id}_{move_number}_{abs(hash(fen)) % 1000000}"

        metadata = {
            "fen": fen,
            "move": move,
            "eval": float(eval_score),
            "game_id": str(game_id),
            "move_number": int(move_number),
            "comment": comment
        }

        self.collection.upsert(
            ids=[pos_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[comment if comment else f"FEN: {fen} | Move: {move}"]
        )

    def add_positions_batch(self, positions_data: List[Dict[str, Any]], batch_size: int = 64):
        """Пакетное добавление позиций для быстрой индексации."""
        total = len(positions_data)
        for i in range(0, total, batch_size):
            batch = positions_data[i:i + batch_size]

            ids: List[str] = []
            embeddings: List[List[float]] = []
            metadatas: List[Dict[str, Any]] = []
            documents: List[str] = []

            for item in batch:
                fen = item["fen"]
                move_num = item.get("move_number", 1)
                game_id = item.get("game_id", "game")

                pos_id = f"{game_id}_{move_num}_{abs(hash(fen)) % 1000000}"
                emb = self.encoder.get_embedding(fen)

                meta = {
                    "fen": fen,
                    "move": item.get("move", ""),
                    "eval": float(item.get("eval", 0.0)),
                    "game_id": str(game_id),
                    "move_number": int(move_num),
                    "comment": item.get("comment", "")
                }

                ids.append(pos_id)
                embeddings.append(emb)
                metadatas.append(meta)
                documents.append(item.get("comment", f"FEN: {fen}"))

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print(f"[+] Загружено позиций: {min(i + batch_size, total)}/{total}")

    def query_similar(
        self,
        fen: str,
        n_results: int = 3,
        exclude_fen: bool = True
    ) -> List[Dict[str, Any]]:
        """Поиск N наиболее структурно близких позиций по косинусному расстоянию."""
        count = self.collection.count()
        if count == 0:
            return []

        query_embedding = self.encoder.get_embedding(fen)
        fetch_k = min(n_results + 2 if exclude_fen else n_results, count)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k
        )

        similar_positions: List[Dict[str, Any]] = []
        if not results["metadatas"] or not results["metadatas"][0]:
            return similar_positions

        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            if exclude_fen and meta.get("fen") == fen:
                continue

            similar_positions.append({
                "fen": meta.get("fen"),
                "move": meta.get("move"),
                "eval": meta.get("eval"),
                "game_id": meta.get("game_id"),
                "comment": meta.get("comment"),
                "distance": round(dist, 4)
            })

            if len(similar_positions) == n_results:
                break

        return similar_positions