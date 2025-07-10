# from .model import CosCNN, MyCustomModel
from .model import CosCNN

MODEL_REGISTRY = {
    'CosCNN': CosCNN,
    # 'MyCustomModel': MyCustomModel,
}