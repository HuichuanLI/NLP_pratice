import json
import math

from segment.data import HMM_POS_ENHANCE_MODEL_PATH
from segment.word_tokenizer.word_based_tokenizer import WordBasedTokenizer


class WordPosTokenizer(WordBasedTokenizer):

    def __init__(self, complex_dict):
        super(WordPosTokenizer, self).__init__(complex_dict)
        self._load_enhance_model()

    def _load_enhance_model(self):
        with open('{}/start_p.json'.format(HMM_POS_ENHANCE_MODEL_PATH)) as file:
            self._hmm_pos_start_p = json.load(file)
        with open('{}/trans_p.json'.format(HMM_POS_ENHANCE_MODEL_PATH)) as file:
            self._hmm_pos_trans_p = json.load(file)
        self._min_start_p = min(self._hmm_pos_start_p.values()) * 10
        self._min_trans_p = min({k: min(v.values()) for k, v in self._hmm_pos_trans_p.items()}.values()) * 10

    def _get_pos(self, word):
        return self.complex_dict.pos(word)

    def _process_all_pos(self, word, is_model_word):
        if is_model_word:
            return word, self._get_pos(word)
        return word, self._get_pos(word)

    def _process_single_pos(self, word, is_model_word):
        if is_model_word:
            return word, self._get_pos(word)[0][0]
        return word, self._get_pos(word)[0][0]

    @staticmethod
    def _post_process(word, pos, enable_offset, start_offset):
        if enable_offset:
            return word, start_offset, start_offset + len(word), pos
        return word, pos

    def _enhance(self, word_pos_pairs):
        if not word_pos_pairs:
            return 0, []
        v = [{}]
        path = {}
        for pos, freq in word_pos_pairs[0][1]:
            v[0][pos] = self._hmm_pos_start_p.get(
                pos, self._min_start_p) + math.log(freq) - self.complex_dict.get_total_freq_log()
            path[pos] = [pos]
        for i in range(1, len(word_pos_pairs)):
            v.append({})
            new_path = {}
            for pos, freq in word_pos_pairs[i][1]:
                (best_prob, best_pos) = max([(v[i - 1][prev_pos] + self._hmm_pos_trans_p.get(prev_pos, {
                    pos: 0
                }).get(pos, self._min_trans_p) + math.log(freq) - self.complex_dict.get_total_freq_log(), prev_pos)
                                             for prev_pos in v[-2]])
                v[i][pos] = best_prob
                new_path[pos] = path[best_pos] + [pos]
            path = new_path
        (best_prob, best_pos) = max((v[-1][pos], pos) for pos, _ in word_pos_pairs[-1][1])
        return best_prob, path[best_pos]

    def pos(self, sentence, start_offset, enable_offset=False, model_seg=None, model_pos_only=None, enhance=True,
            use_ner=False):
        if model_pos_only:
            raise ValueError('Does not support separate models for segment and pos for now')
        else:
            if not enhance:
                for word, pos in self._seg(sentence, model_seg=model_seg, pos_enhance=self._process_single_pos,
                                           use_ner=use_ner):
                    yield self._post_process(word, pos, enable_offset, start_offset)
                    start_offset += len(word)
            else:
                word_pos_pairs = list(
                    self._seg(sentence, model_seg=model_seg, pos_enhance=self._process_all_pos, use_ner=use_ner))
                best_prob, path = self._enhance(word_pos_pairs)
                for i, pos in enumerate(path):
                    yield self._post_process(word_pos_pairs[i][0], pos, enable_offset, start_offset)
                    start_offset += len(word_pos_pairs[i][0])
