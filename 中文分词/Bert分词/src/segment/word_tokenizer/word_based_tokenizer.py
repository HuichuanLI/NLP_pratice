import math

from segment.word_tokenizer.word_graph import WordGraph, Node


class WordBasedTokenizer(object):
    """词图分词器基类"""

    def __init__(self, complex_dict):
        super(WordBasedTokenizer, self).__init__()
        self.complex_dict = complex_dict
        self._ner = None

    def _find_all_words(self, sentence, sentence_length, seg_all):
        graph = WordGraph()
        all_words = {0: [0], sentence_length - 1: [sentence_length - 1]}
        graph.insert_start_word(WordGraph.NODE_S)
        for i in range(1, sentence_length - 1):
            word_end_index_list = [i]
            word = sentence[i]
            freq = self.complex_dict.freq(sentence[i]) or 1
            graph.insert_start_word(Node(word, self._calculate_weight(freq), freq=freq))
            j = i + 1
            fragment = ''.join(sentence[i:j + 1])
            while j < sentence_length - 1:
                source = self.complex_dict.is_in(fragment)
                source = self._check_is_model_word(i, j, source)
                source = self._check_is_ner_word(i, j, source)
                if not source:
                    break
                freq = self.complex_dict.freq(fragment)
                if freq:
                    word_end_index_list.append(j)
                    graph.insert_start_word(Node(fragment, self._calculate_weight(freq), source, freq))
                j += 1
                char_list = sentence[i:j + 1]
                if seg_all and len(char_list) >= len(sentence) - 2:
                    break
                fragment = ''.join(char_list)
            all_words[i] = word_end_index_list
        return graph, all_words

    def _check_is_model_word(self, i, j, source):
        if source == 'model_word_dict' and not self.complex_dict.get_model_dict().is_model_seg(i, j + 1):
            source = None
        return source

    def _check_is_ner_word(self, i, j, source):
        if source == 'ner_word_dict' and not self.complex_dict.get_ner_dict().is_ner_seg(i, j + 1):
            source = None
        return source

    def _make_word_graph(self, sentence, sentence_length, seg_all):
        graph, all_words = self._find_all_words(sentence, sentence_length, seg_all)
        for i in range(sentence_length - 1):
            for end_words in self._find_end_words_list(i, all_words):
                graph.insert_end_words(end_words)
        return graph

    @staticmethod
    def _find_end_words_list(start_char_index, all_words):
        for end_index in all_words[start_char_index]:
            base_index = sum([len(all_words[i]) for i in range(end_index + 1)])
            end_words = [base_index + count for count in range(len(all_words[end_index + 1]))]
            yield end_words

    def _seg(self, sentence, model_seg=None, pos_enhance=None, seg_all=False, regex_level=0, use_ner=False):
        full_match = self._match_regex_word(sentence, regex_level)
        if seg_all and full_match:
            return
        self.complex_dict.get_model_dict().clear()
        self.complex_dict.get_ner_dict().clear()
        if model_seg:
            model_words = model_seg(sentence)
            self.complex_dict.get_model_dict().load_model_words(model_words)
        if use_ner:
            ner_words = self._ner_find(sentence)
            self.complex_dict.get_ner_dict().load_ner_words(ner_words)
        sentence = [WordGraph.NODE_S.key] + list(sentence) + [WordGraph.NODE_E.key]
        sentence_length = len(sentence)
        graph = self._make_word_graph(sentence, sentence_length, seg_all)
        route = graph.calculate()
        index = route[0][1]
        node = graph.get_node(index)
        while node != WordGraph.NODE_E:
            word = node.key
            if pos_enhance is None:
                yield word
            else:
                yield pos_enhance(word, node.source == 'model_word_dict' or node.source == 'ner_word_dict')
            index = route[index][1]
            node = graph.get_node(index)

    def _ner_find(self, sentence):
        # TODO
        return []

    def _calculate_weight(self, freq):
        return math.log(freq or freq + 1) - self.complex_dict.get_total_freq_log()

    def _match_regex_word(self, content, regex_level=0):
        self.complex_dict.get_regex_dict().clear()
        full_match = False
        regex_word_pairs = self.complex_dict.get_regex_dict().get_regex_word_pairs()
        for level, regex, word, pos, freq in regex_word_pairs:
            if level >= regex_level:
                for m in regex.finditer(content):
                    if m.end() - m.start() == len(content):
                        full_match = True
                    word = content[m.start():m.end()]
                    self.complex_dict.get_regex_dict().add_word(word, pos, freq)
        return full_match
