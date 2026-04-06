from torchvision.transforms import Compose

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registered_builders = dict()

    def build(self, config) -> object:
        assert len(config.keys()) == 1
        name = list(config.keys())[0]
        return self._registered_builders[name](config[name])

    def register(self, name):
        def _register(_cls):
            self.register_builder(name)(_cls.from_config)
            return _cls
        return _register
    
    def register_builder(self, name):
        assert name not in self._registered_builders.keys(), f'"{name}" is already registered'
        def _register_builder(_builder):
            self._registered_builders[name] = _builder
            return _builder
        return _register_builder


GROUP = Registry('group')

MODULE = Registry('module')

OPTIMIZER = Registry('optimizer')

LOSS = Registry('loss')

DATASET = Registry('dataset')

class TransformRegistry(Registry):
    def build(self, config):
        if type(config) is list:
            return Compose([self.build(c) for c in config])
        else:
            assert len(config.keys()) == 1
            name = list(config.keys())[0]
            return self._registered_builders[name](config[name])

TRANSFORM = TransformRegistry('transform')
