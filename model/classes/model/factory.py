from model.classes.model.pix2code import pix2code
from model.classes.model.pix2codebilstm import pix2codeBiLSTM


class ModelFactory:
	@staticmethod
	def create_model(model, input_shape, output_size, output_path):
		if model == 'base':
			return pix2code(input_shape, output_size, output_path)

		elif model == 'bilstm':
			return pix2codeBiLSTM(input_shape, output_size, output_path)

		# elif name == 'GRU':
		#     return gru(input_shape, output_size, output_path)
