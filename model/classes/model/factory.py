from classes.model.pix2code import pix2code
from classes.model.pix2codebilstm import pix2codeBiLSTM
from classes.model.pix2codegru import pix2codegru


class ModelFactory:
    @staticmethod
    def create_model(model, input_shape, output_size, output_path, encoding_type):
        if model == 'base':
            return pix2code(input_shape, output_size, output_path, encoding_type)
        elif model == 'bilstm':
            return pix2codeBiLSTM(input_shape, output_size, output_path, encoding_type)
        elif model == 'gru':
            return pix2codegru(input_shape, output_size, output_path, encoding_type)
        else:
            raise Exception("Model not found")
