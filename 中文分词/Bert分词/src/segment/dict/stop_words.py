class StopWords(object):

    def __init__(self, path=None):
        super(StopWords, self).__init__()
        self._stop_words = set()
        self._path = path
        if self._path:
            self._load()

    def _load(self):
        with open(self._path, 'r', encoding='utf-8') as file:
            for line in file:
                self._stop_words.add(line.strip())

    def is_in(self, word):
        return word in self._stop_words
