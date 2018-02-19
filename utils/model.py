import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1,floatX=float32,cnmem=0.8"
# os.system("THEANO_FLAGS='device=gpu1'")

import datetime
import numpy as np
from time import time
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Sequential
#from keras.layers import Input, concatenate
from keras.models import Model as keras_Model
from keras.optimizers import Adam
#from keras.layers.merge import Add, Average

from image_proc import binarize, crop_resize
from evaluation import dice_coef_loss, dice_coef, recall
from pre_train import data_augmentation, pre_reshape
#from utils.src.models.post_train import post_reshape
from interfaces import Model

METRICS = {
    'dice': dice_coef,
    'recall': recall,
    # 'accuracy': metrics.accuracy_score,
    # 'precision': metrics.precision_score,
    # 'f1': metrics.f1_score,
}


class BaseModel(Model):
    def __init__(self, batch_size, input_row, input_col, output_row, output_col, objective, metrics=['binary_accuracy'], data_augmentation_kwargs=None, percentage_x=1, percentage_y=1, weights_path=None):
        self.batch_size = batch_size
        self.input_col = input_col
        self.input_row = input_row
        self.output_col = output_col
        self.output_row = output_row
        self.input_shape = (1, self.input_row, self.input_col)
        self.output_shape = (1, self.output_row, self.output_col)
        self.percentage_x = percentage_x
        self.percentage_y = percentage_y
        self.objective = objective
        self.metrics = metrics
        self.data_augmentation_kwargs = data_augmentation_kwargs

        if self.objective == 'dice':
            self.objective = dice_coef_loss

        self.metrics = map(lambda x: METRICS[x] if x in METRICS else x, self.metrics)
        
        self.weights_path = weights_path

    @property
    def model(self):
        # TODO
        return self.__model

    def fit(self, x_train, y_train, x_validation, y_validation, nb_epoch=100, len_epoch=None, best_filepath=None, callbacks=[]):
        """
        Fit model using the parameters passed to the method.

        model: model to be fitted
        x_train: training input in np array
        y_train: training label
        batch_size: batch_size
        nb_epoch: number of epochs
        x_validation: validation input
        y_validation: validation label
        best_filepath: if you want to save the best model monitoring val_loss, put the filepath here
        callbacks: any new callback you want to use
        """
        if len_epoch is None:
            try:
                len_epoch = len(x_train)
            except:
                len_epoch = x_train.shape[0]

        # Resize and reshape series
        x_train = self.crop_borders(x_train)
        x_validation = self.crop_borders(x_validation)
        y_train = self.crop_borders(y_train)
        y_validation = self.crop_borders(y_validation)

        x_train = self.resize_series(x_train, self.input_row, self.input_col)
        x_validation = self.resize_series(x_validation, self.input_row, self.input_col)
        y_train = self.resize_series(y_train, self.output_row, self.output_col, interp='nearest')
        y_validation = self.resize_series(y_validation, self.output_row, self.output_col, interp='nearest')
        x_train, y_train, x_validation, y_validation = self.reshape(x_train, y_train, x_validation, y_validation)

        generator = None
        if self.data_augmentation_kwargs:
            generator = data_augmentation(x_train, y_train, **self.data_augmentation_kwargs)

        if best_filepath:
            checkpointer = ModelCheckpoint(filepath=best_filepath, verbose=1, save_best_only=True)
            callbacks.append(checkpointer)

        if generator:
            self.model.fit_generator(generator, samples_per_epoch=len_epoch, nb_epoch=nb_epoch, validation_data=(x_validation, y_validation), callbacks=callbacks)
        else:
            self.model.fit({'main_input': x_train, 'left_input': np.roll(x_train,1), 'right_input': np.roll(x_train,-1)},
                           {'main_output': y_train, 'left_output': np.roll(y_train,1), 'right_output': np.roll(y_train,-1)}, 
                           batch_size=self.batch_size, epochs=nb_epoch,
                           validation_data=({'main_input': x_validation, 'left_input': np.roll(x_validation,1), 'right_input': np.roll(x_validation,-1)}, {'main_output': y_validation, 'left_output': np.roll(y_validation,1), 'right_output': np.roll(y_validation,-1)}), callbacks=callbacks)

        return y_validation, self.model.predict(x_validation)

    def crop_borders(self, serie):

        croped_serie = []
        for img in serie:
            assert (self.percentage_x < 1) and (self.percentage_y < 1), "Percentages to crop borders should be smaller than one."
            x = img.shape[0]
            y = img.shape[1]

            fat_x = np.ceil(self.percentage_x * x)
            fat_y = np.ceil(self.percentage_y * y)

            crop_img = img[int(fat_x): int(x - fat_x + 1), int(fat_y): int(y - fat_y + 1)]

            croped_serie.append(crop_img)

        return croped_serie

    def resize_series(self, serie, rows, cols, interp='bilinear'):
        """
        Resize all images in all cases to rows, cols.

        series: array of numpy 3d images
        rows: size of desired row
        cols: size of desired collumms
        TODO: couldn't manage to do this in place, will return to that after the necessity arises
        """
        resize_serie = []
        for img in serie:
            resize_serie.append(crop_resize(img, rows, cols, interp))
        return resize_serie

    def reshape(self, *args):
        # IDK if it works!!!
        args = list(args)
        for i in range(len(args)):
            args[i] = pre_reshape(args[i])

        return tuple(args)

    def predict(self, data):
        """
        Predict labels for input data.

        Data must be reshaped to (samples, channels, rows, cols) if backend is Theano
        and (samples, rows, cols, channels) if backend is TensorFlow.

        Output is reshaped to a list of binary masks = ([mask0], [mask1], [mask], ...)
        """
        # 1 - Starting time
        start = time()
        print ''
        print 'PREDICT'
        print '-----------------------------------------------------------'

        # 2 - Reshape, predict and binarize
        # data_predict = list(self.reshape(data))
        data_predict = data
        y = binarize(self.model.predict(data_predict, batch_size=self.batch_size, verbose=1))

        # 3 - Time elapsed
        time_elapsed = time() - start
        if time_elapsed > 60:
            time_elapsed /= 60

        # 2 - Final messages
        print ''
        print '> Time elapsed:', time_elapsed
        print '-----------------------------------------------------------'

        return y

    def predict_proba(self, data):
        """
        Predict labels probabilities for input data.

        Data must be reshaped to (samples, channels, rows, cols) if backend is Theano
        and (samples, rows, cols, channels) if backend is TensorFlow.

        Output is reshaped to a list = ([output_img0], [output_img1], [output_img2], ...)
        """
        data_predict = list(self.reshape(data))
        y = self.model.predict_proba(data_predict, batch_size=self.batch_size, verbose=1)

        return post_reshape(y)

    def save(self, model, path, idx=None, file_name=None):
        return self.__save_keras_model(model=model, path=path, idx=idx, file_name=file_name)

    def load(self, model_idx, path='models', model_name=None, custom_objects=None):
        # custom_objects_dict = {

        # }
        if not hasattr(self, '__model'):
            model_idx_list = [model_idx]
            model = self.__load_keras_model(model_idx_list=model_idx_list, path=path, model_name=model_name, custom_objects=custom_objects)[0]
            self.__model = model
        else:
            print 'Model attribute is already set!'

        return self.__model

    def __save_keras_model(self, path, model, idx=None, file_name=None):
        """
        TODO - UPDATE

        Saving Keras models on the disk in .h5 format
        Params:
        path > path to model folder
        model -> keras model to be saved
        idx -> index from KFold. When not using CV, the default index is zero
        file_name (optional) -> a name suggestion for the model

        If file_name is None, the model is named with the python script name
        If one is trying to save the model from a ipynb, the default name is the datetime

        """
        if not file_name:
            try:
                #  Get hyp name
                work_dir = os.getcwd().split('/')
                file_name = list(reversed(work_dir))[0]
                # file_name = os.path.basename(__file__).split(".py")[0]
            except:
                file_name = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

        if idx:
            model_name = '{}_model_{}.h5'.format(file_name, idx)
        else:
            model_name = '{}_model.h5'.format(file_name)

        if os.path.isfile(os.path.join(path, model_name)):
            date = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
            model_name = model_name + '({})'.format(date)

        print ('Saving model... [{}]'.format(model_name))
        model.save(os.path.join(path, model_name))

    def __load_keras_model(self, model_idx_list=[], path='models', model_name=None, custom_objects=None):
        '''
        Method that loads Keras model from disk, saved in .h5 format.

        Parameters
        ----------
        model_idx_list : int array
            Array with the index of the models to be loaded

        path : string
            Path to models folder. (Default='models')

        model_name : string
            Model name if one wants to use an alternative model name instead of the hyp name.

        custom_objects : dict
            Custom loss functions used to train model. (Default=None)
                ex: {'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}

        Returns
        -------

        Returns an array from loaded Keras models.

        Notes
        -----
        !!!
        It's highly recommend to use the convension to name the models!
        Convension: <hyp name>_model_<idx>.h5
        !!!

        If no model_name is passed, it takes the name of the hyp, from the working directory, as the convension above recommends.

        '''
        # 0 - Starting time
        start = time()
        print ''
        print 'LOAD KERAS MODEL'
        print '-----------------------------------------------------------'

        # 1 - Get model name
        if not model_name:
            #  Get hyp name
            work_dir = os.getcwd().split('/')
            model_name = list(reversed(work_dir))[0]  # Model name == Hyp Name

        # 2 - Check if model index array is not empty
        if not len(model_idx_list):
            raise Exception('Array of model index is empty!')

        # 3 - Loop for loading models
        model_arr = []
        for i in model_idx_list:

            #  3a - Get model_file (path + model file name)
            f_name = '{}_model_{}.h5'.format(model_name, i)
            model_file = os.path.join(path, f_name)

            #  3b - Check if model exists
            if not os.path.isfile(model_file):
                raise Exception('Model {} does not exist or path is wrong!'.format(model_file))

            print 'Loading ... {}'.format(model_file)
            #  3c - Load ...
            print custom_objects
            if custom_objects:
                model_arr.append(load_model(model_file, custom_objects=custom_objects))
            else:
                print model_file
                model_arr.append(load_model(model_file))

        # 4 - Time elapsed
        time_elapsed = time() - start
        time_units = 'seconds'
        if time_elapsed > 60:
            time_elapsed /= 60
            time_units = 'minutes'

        # 5 - Final messages
        print ''
        print '> {} Keras model(s) loaded!'.format(len(model_arr))
        print '> Time elapsed:', time_elapsed, time_units
        print '-----------------------------------------------------------'

        return model_arr
    
    
class UNetModel(BaseModel):

    def load_weights(self, path):
        self.model.load_weights(path)
        return

    def center_normalize(self, x):
        """Custom activation for online sample-wise center and std. normalization."""
        return (x - K.mean(x)) / K.std(x)

    @property
    def model(self):
        """
        Return model used to win the sciencebowl.

        The first layer was changed to get a custom normalization
        input_shape: input shape
        objective: loss function
        metrics: metric used to evaluate correctness
        """
        if not hasattr(self, '__model'):
            
            left_input = Input(shape=self.input_shape, name='left_input')
            main_input = Input(shape=self.input_shape, name='main_input')            
            right_input = Input(shape=self.input_shape, name='right_input')
            
            # Ramo da esquerda
            
            left = Activation(activation=self.center_normalize)(left_input)
            #Area do 128
            left1 = Conv2D(16, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(left) 
            left = Activation(activation='relu')(left1)
            left = BatchNormalization()(left)
            left = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(left)            
            #Area do 64
            left2 = Conv2D(32, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(left)            
            left = Activation(activation='relu')(left2)
            left = BatchNormalization()(left)
            left = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(left)            
            #Area do 32            
            left3 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(left)            
            left = Activation(activation='relu')(left3)
            left = BatchNormalization()(left)
            left = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(left) 
            #Area do 16
            left4 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(left)            
            left = Activation(activation='relu')(left4)
            left = BatchNormalization()(left)
            left5 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(left)            
            left = Activation(activation='relu')(left5)
            left = BatchNormalization()(left)
            left = UpSampling2D(size=(2, 2), data_format="channels_first")(left)
            #Area do 32            
            left6 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(left) 
            left = Activation(activation='relu')(left6)
            left = BatchNormalization()(left)
            left = UpSampling2D(size=(2, 2), data_format="channels_first")(left)
            #Area do 64    
            left7 = Conv2D(32, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(left)
            left = Activation(activation='relu')(left7)
            left = BatchNormalization()(left)
            left = UpSampling2D(size=(2, 2), data_format="channels_first")(left)
            #Area do 128
            left8 = Conv2D(16, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(left)
            left = Activation(activation='relu')(left8)
            left = BatchNormalization()(left)
            left9 = Convolution2D(8, (7, 7), padding='same', activation='linear', kernel_initializer='he_normal')(left)
            left = Activation(activation='relu')(left9)
            left = BatchNormalization()(left)
            left_output = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal', name='left_output')(left)            
            
            # Ramo da direita
            
            right = Activation(activation=self.center_normalize)(right_input)
            #Area do 128
            right1 = Conv2D(16, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(right) 
            right = Activation(activation='relu')(right1)
            right = BatchNormalization()(right)
            right = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(right)            
            #Area do 64
            right2 = Conv2D(32, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(right)            
            right = Activation(activation='relu')(right2)
            right = BatchNormalization()(right)
            right = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(right)            
            #Area do 32            
            right3 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(right)            
            right = Activation(activation='relu')(right3)
            right = BatchNormalization()(right)
            right = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(right) 
            #Area do 16
            right4 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(right)            
            right = Activation(activation='relu')(right4)
            right = BatchNormalization()(right)
            right5 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(right)            
            right = Activation(activation='relu')(right5)
            right = BatchNormalization()(right)
            right = UpSampling2D(size=(2, 2), data_format="channels_first")(right)
            #Area do 32            
            right6 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(right) 
            right = Activation(activation='relu')(right6)
            right = BatchNormalization()(right)
            right = UpSampling2D(size=(2, 2), data_format="channels_first")(right)
            #Area do 64    
            right7 = Conv2D(32, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(right)
            right = Activation(activation='relu')(right7)
            right = BatchNormalization()(right)
            right = UpSampling2D(size=(2, 2), data_format="channels_first")(right)
            #Area do 128
            right8 = Conv2D(16, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(right)
            right = Activation(activation='relu')(right8)
            right = BatchNormalization()(right)
            right9 = Convolution2D(8, (7, 7), padding='same', activation='linear', kernel_initializer='he_normal')(right)
            right = Activation(activation='relu')(right9)
            right = BatchNormalization()(right)
            right_output = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal', name='right_output')(right)  
            
            # Ramo central
            
            main = Activation(activation=self.center_normalize)(main_input)
            #Area do 128
            main1 = Conv2D(16, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(main)            
            main1 = Average()([left1, right1, main1])
            main1 = Activation(activation='relu')(main1)
            main = BatchNormalization()(main1)
            main = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(main)            
            #Area do 64
            main2 = Conv2D(32, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(main)    
            main2 = Average()([left2, right2, main2])
            main2 = Activation(activation='relu')(main2)        
            main = BatchNormalization()(main2)
            main = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(main)            
            #Area do 32            
            main3 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(main)   
            main3 = Average()([left3, right3, main3])
            main3 = Activation(activation='relu')(main3)
            main = BatchNormalization()(main3)
            main = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(main) 
            #Area do 16
            main4 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(main)    
            main4 = Average()([left4, right4, main4])
            main4 = Activation(activation='relu')(main4)
            main = BatchNormalization()(main4)
            main5 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(main)    
            main5 = Average()([left5, right5, main5])
            main5 = Activation(activation='relu')(main5)
            main = BatchNormalization()(main5)
            main = UpSampling2D(size=(2, 2), data_format="channels_first")(main)
            #Area do 32            
            main6 = Conv2D(64, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(main) 
            main6 = Average()([left6, right6, main6])
            main6 = Activation(activation='relu')(main6)
            main = BatchNormalization()(main6)
            main = UpSampling2D(size=(2, 2), data_format="channels_first")(main)
            #Area do 64    
            main7 = Conv2D(32, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(main)
            main7 = Average()([left7, right7, main7])
            main7 = Activation(activation='relu')(main7)
            main = BatchNormalization()(main7)
            main = UpSampling2D(size=(2, 2), data_format="channels_first")(main)
            #Area do 128
            main8 = Conv2D(16, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(main)
            main8 = Average()([left8, right8, main8])
            main8 = Activation(activation='relu')(main8)
            main = BatchNormalization()(main8)
            main9 = Convolution2D(8, (7, 7), padding='same', activation='linear', kernel_initializer='he_normal')(main)
            main9 = Average()([left9, right9, main9])
            main9 = Activation(activation='relu')(main9)
            main = BatchNormalization()(main9)
            main_output = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal', name='main_output')(main)
            
            model = keras_Model(inputs=[main_input, left_input, right_input], outputs=[main_output, left_output, right_output])
            
            if self.weights_path:
                model.load_weights(self.weights_path)

            # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=Adam(lr=1e-5), loss=self.objective, metrics=self.metrics)
            
            # model.compile(optimizer=Adam(lr=1e-5), loss=self.objective, metrics=['accuracy'])
            self.__model = model

        return self.__model