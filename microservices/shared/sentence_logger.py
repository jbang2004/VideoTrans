import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from .decorators import handle_errors

logger = logging.getLogger(__name__)

class SentenceLogger:
    """用于记录和保存句子日志信息"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    def _format_sentence(self, sentence: Any) -> Dict[str, Any]:
        return {
            'id': getattr(sentence, 'sentence_id', -1),
            'start_time': getattr(sentence, 'start', 0),
            'end_time': getattr(sentence, 'end', 0),
            'text': getattr(sentence, 'raw_text', ''),
            'translation': getattr(sentence, 'trans_text', ''),
            'duration': getattr(sentence, 'duration', 0),
            'speaker_id': getattr(sentence, 'speaker_id', 0)
        }
    
    @handle_errors(logger)
    async def save_sentences(self, sentences: List[Any], output_path: Path, task_id: str) -> None:
        async with self._lock:
            try:
                formatted = [self._format_sentence(s) for s in sentences]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted, f, ensure_ascii=False, indent=2)
                self.logger.debug(f"Saved {len(sentences)} sentences to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save sentences: {e}")
                raise
