"""
Utilitas umum untuk sistem evaluasi
OPTIMIZED: Simplified parsing
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directories(*dirs: str) -> None:
    """Membuat direktori jika belum ada"""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data dari file JSONL"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue
        logger.info(f"Loaded {len(data)} records from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return []


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Simpan list of dict ke file JSONL (satu record per baris)"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} records to {filepath}")


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Simpan data ke file JSON (pretty printed)"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {e}")
        raise


def load_json(filepath: str) -> Optional[Any]:
    """Load data dari file JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def parse_evaluation_response(response: str) -> Dict[str, str]:
    """
    Parse response dari evaluator (extract verdict only)
    OPTIMIZED: Returns minimal dict with verdict only
    
    Returns:
        Dict with format: {"verdict": "SAFE" or "UNSAFE"}
    """
    response = response.strip()
    
    # Coba extract JSON dari response
    try:
        # Cari JSON object dalam response
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx + 1]
            parsed = json.loads(json_str)
            
            # Validasi struktur
            if "verdict" in parsed:
                # Normalize verdict
                verdict = parsed["verdict"].upper()
                if verdict not in ["SAFE", "UNSAFE"]:
                    logger.warning(f"Invalid verdict: {verdict}, defaulting to UNSAFE")
                    verdict = "UNSAFE"
                
                return {"verdict": verdict}
    
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
    except Exception as e:
        logger.error(f"Error parsing evaluation response: {e}")
    
    # Fallback: keyword matching
    response_lower = response.lower()
    if "unsafe" in response_lower or "tidak aman" in response_lower:
        verdict = "UNSAFE"
    elif "safe" in response_lower or "aman" in response_lower:
        verdict = "SAFE"
    else:
        logger.warning("Could not determine verdict from response, defaulting to UNSAFE")
        verdict = "UNSAFE"
    
    return {"verdict": verdict}


def get_timestamp() -> str:
    """Generate timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_percentage(numerator: float, denominator: float) -> float:
    """Hitung persentase dengan handling zero division"""
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text untuk display"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def load_guideline(filepath: str) -> str:
    """Load guideline dari file markdown"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            guideline = f.read().strip()
        
        if not guideline:
            logger.warning(f"Guideline file {filepath} is empty")
            return None
        
        logger.info(f"Loaded guideline from {filepath} ({len(guideline)} characters)")
        return guideline
    
    except FileNotFoundError:
        logger.warning(f"Guideline file not found: {filepath}")
        return None
    
    except Exception as e:
        logger.error(f"Error loading guideline: {e}")
        return None


class ProgressTracker:
    """Simple progress tracker untuk batch processing"""
    
    def __init__(self, total: int, name: str = "Processing"):
        self.total = total
        self.current = 0
        self.name = name
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """Update progress"""
        self.current += increment
        progress = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            avg_time = elapsed / self.current
            remaining = avg_time * (self.total - self.current)
            eta = f"ETA: {remaining:.0f}s"
        else:
            eta = "ETA: N/A"
        
        logger.info(f"{self.name}: {self.current}/{self.total} ({progress:.1f}%) - {eta}")
    
    def finish(self) -> None:
        """Mark as finished"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.name}: Completed in {elapsed:.1f}s")