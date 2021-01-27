import tensorflow as tf
from functools import lru_cache
from ..polisci_helpers import get_polisci_path


@lru_cache(maxsize=2048)
def load_polisci_model(filename) -> tf.keras.Model:
    path = get_polisci_path('models', filename)
    return tf.keras.models.load_model(path)



__all__ = ['load_polisci_model']
