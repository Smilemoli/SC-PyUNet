# model_config.py

class ModelConfig:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 16
        self.num_epochs = 50
        self.input_size = (224, 224)
        self.num_classes = 2
        self.encoder = 'swin_transformer'
        self.decoder = 'convnext'
        self.model_name = 'SwinConvNextUNet'