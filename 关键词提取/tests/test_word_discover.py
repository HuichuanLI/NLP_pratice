import unittest

from keywords.word_discover import WordDiscover


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        data_path = 'tests/test_data/sports_1000.txt'
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(line)

    def test_something(self):
        word_discover = WordDiscover()
        r = word_discover.discover(self.samples)
        print(r)


if __name__ == '__main__':
    unittest.main()
