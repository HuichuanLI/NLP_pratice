import abc


class BaseDict(object, metaclass=abc.ABCMeta):
    """词典基类"""

    @abc.abstractmethod
    def is_in(self, word):
        pass

    @abc.abstractmethod
    def freq(self, word):
        pass

    @abc.abstractmethod
    def pos(self, word):
        pass

    @abc.abstractmethod
    def get_total_freq_log(self):
        pass
