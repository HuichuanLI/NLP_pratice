import unittest
from metrics import EntityScore
import json


class MyTestCase(unittest.TestCase):
    def test_single_entity_score(self):
        tags_actual = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'B-PER.NAM', 'I-PER.NAM', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],  # 1个实体
            ['O', 'B-PER.NAM', 'I-PER.NAM', 'I-PER.NAM', 'O', 'O', 'B-PER.NAM', 'I-PER.NAM', 'O', 'O']  # 2个实体
        ]
        tags_predict = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'B-PER.NAM', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],  # 错1个
            ['O', 'B-PER.NAM', 'I-PER.NAM', 'I-PER.NAM', 'O', 'O', 'O', 'O', 'O', 'O']  # 漏1个
        ]
        r = EntityScore.single_entity_score(tags_predict, tags_actual)
        print(r)

    def test_multi_entities_score(self):
        tags_actual = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'B-LOC.NAM', 'I-LOC.NAM', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],  # 1个LOC
            ['O', 'B-PER.NAM', 'I-PER.NAM', 'I-PER.NAM', 'O', 'O', 'B-PER.NAM', 'I-PER.NAM', 'O', 'O']  # 2个PER
        ]
        tags_predict = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'B-LOC.NAM', 'I-LOC.NAM', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],  # LOC对1个
            ['O', 'B-PER.NAM', 'I-PER.NAM', 'I-PER.NAM', 'O', 'O', 'O', 'O', 'O', 'O']  # PER漏1个，对1个
        ]
        r = EntityScore.multi_entities_score(tags_predict, tags_actual, entity_types=('PER', 'LOC'))
        print(json.dumps(r, indent=4))


if __name__ == '__main__':
    unittest.main()
