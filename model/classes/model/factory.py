from classes.model.pix2code import pix2code
from classes.model.pix2codeResNet import pix2codeResNet
from classes.model.pix2codebilstm import pix2codeBiLSTM
from classes.model.pix2codegru import pix2codegru
from classes.model.pix2codeVGG16 import pix2codeVGG16


class ModelFactory:
    @staticmethod
    def create_model(model: str, input_shape, output_size, output_path, encoding_type):
        if model == pix2code.name:
            return pix2code(input_shape, output_size, output_path, encoding_type)
        elif model == pix2codeBiLSTM.name:
            return pix2codeBiLSTM(input_shape, output_size, output_path, encoding_type)
        elif model == pix2codegru.name:
            return pix2codegru(input_shape, output_size, output_path, encoding_type)
        elif model == pix2codeResNet.name:
            return pix2codeResNet(input_shape, output_size, output_path, encoding_type)
        elif model == pix2codeVGG16.name:
            return pix2codeVGG16(input_shape, output_size, output_path, encoding_type)
        else:
            raise Exception("Model not found")
