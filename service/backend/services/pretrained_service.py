import joblib
from pathlib import Path
from typing import List, Optional, Dict, Any
import pickle

from service.backend.core.config import MODEL_STORAGE, logger
from service.backend.models.schemas import PretrainedModelInfo


class PretrainedModelService:
    
    def __init__(self):
        self.model_storage = Path(MODEL_STORAGE)
        self.loaded_pretrained_models: Dict[str, Dict[str, Any]] = {}
    
    def scan_pretrained_models(self) -> List[PretrainedModelInfo]:
        """Сканирует директорию saved_models и возвращает список найденных моделей."""
        models = []
        
        if not self.model_storage.exists():
            logger.warning(f"Директория {self.model_storage} не существует")
            return models
        
        for file_path in self.model_storage.glob("*.pkl"):
            try:
                stat = file_path.stat()
                
                model_info = PretrainedModelInfo(
                    filename=file_path.name,
                    model_id=file_path.stem,
                    file_size=stat.st_size,
                    is_loaded=file_path.name in self.loaded_pretrained_models
                )
                models.append(model_info)
                logger.info(f"Найдена предобученная модель: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {str(e)}")
        
        return models
    
    def load_pretrained_model(self, filename: str) -> Dict[str, Any]:
        """Загружает предобученную модель из файла."""
        file_path = self.model_storage / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл модели {filename} не найден")
        
        if not filename.endswith('.pkl'):
            raise ValueError("Поддерживаются только файлы .pkl")
        
        try:
            model_data = joblib.load(file_path)
            logger.info(f"Предобученная модель {filename} успешно загружена")
            
            self.loaded_pretrained_models[filename] = model_data
            
            return model_data
            
        except AttributeError as e:
            if "_RemainderColsList" in str(e) or "sklearn" in str(e):
                logger.warning(f"Обнаружена проблема совместимости sklearn для {filename}: {str(e)}")
                return self._try_compatibility_loading(file_path, filename)
            else:
                logger.error(f"Ошибка атрибутов при загрузке модели {filename}: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {filename}: {str(e)}")
            raise
    
    def _try_compatibility_loading(self, file_path: Path, filename: str) -> Dict[str, Any]:
        try:
            logger.info(f"Попытка загрузки {filename} через pickle...")
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            logger.info(f"Модель {filename} загружена через pickle")
            self.loaded_pretrained_models[filename] = model_data
            return model_data
            
        except Exception as pickle_error:
            logger.warning(f"Стандартный pickle не сработал: {str(pickle_error)}")
            
    def get_loaded_model(self, filename: str) -> Optional[Dict[str, Any]]:
        """Возвращает загруженную модель, если она есть в памяти."""
        return self.loaded_pretrained_models.get(filename)
    
    def unload_model(self, filename: str) -> bool:
        """Выгружает модель из памяти."""
        if filename in self.loaded_pretrained_models:
            del self.loaded_pretrained_models[filename]
            logger.info(f"Модель {filename} выгружена из памяти")
            return True
        return False
    

pretrained_service = PretrainedModelService()
