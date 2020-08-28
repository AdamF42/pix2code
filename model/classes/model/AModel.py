__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from keras.models import model_from_json

from classes.TimeHistory import TimeHistory
from keras.optimizers import RMSprop


class AModel:
    def __init__(self, input_shape, output_size, output_path, encoding_type):
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_path = output_path
        self.name = ""
        self.callback = TimeHistory()
        self.encoding_type = encoding_type

    def save(self):
        model_json = self.model.to_json()
        with open("{}/{}_{}.json".format(self.output_path, self.name, self.encoding_type), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("{}/{}_{}.h5".format(self.output_path, self.name, self.encoding_type))

    def load(self, name=""):
        output_name = self.name if name == "" else name
        with open("{}/{}_{}.json".format(self.output_path, output_name, self.encoding_type), "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("{}/{}_{}.h5".format(self.output_path, output_name, self.encoding_type))

    def get_time_history(self):  # TODO: fix time
        return self.callback.times

    def minevaluate(self, dataset):
        dataleng = len(dataset.input_images)
        batch_size = 2048
        step = dataleng // batch_size + 1
        sumscore = 0.0
        sumloss = 0.0
        for i in range(step):
            images, partial_captions, next_words = dataset.minconvert_arrays(i, batch_size)
            loss, score = self.model.evaluate([images, partial_captions], next_words, batch_size=64)
            sumscore = sumscore + score
            sumloss = sumloss + loss
            print("loss:", loss, "accuracy:", score)
        avscore = sumscore / step
        avloss = sumloss / step
        return avscore, avloss

    def compile(self):
        optimizer = RMSprop(lr=0.0002, clipvalue=1.0)
        # optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
