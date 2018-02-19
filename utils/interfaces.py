# coding: utf-8


class PreProc(object):
    def __init__(self, *args, **kwargs):
        pass

    def process(self, imgs):
        raise NotImplementedError('not implemented pre process')


class Model(object):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data, labels):
        raise NotImplementedError('not implemented fit')

    def get_internal_model(self):
        return self.__model

    def __fit_entire_model(self):
        return 'x'

    def predict(self, data):
        raise NotImplementedError('not implemented predict')


class PostProc(object):
    def __init__(self, *args, **kwargs):
        pass

    def process(self, imgs):
        raise NotImplementedError('not implemented pos process')


class Evaluate(object):
    def __init__(self, *args, **kwargs):
        pass

    def eval(self, predict, target, predict_proba=None):
        raise NotImplementedError('not implemented evaluation')


class Visualize(object):
    def __init__(self, *args, **kwargs):
        pass

    def show(self, predict, target, predict_proba=None):
        raise NotImplementedError('not implemented visualize data')
