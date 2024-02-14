MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")


class Viterbi(object):

    def __init__(self, tags, start_p, trans_p, prev_tags, start_tags, end_tags):
        self._tags = tags
        self._start_p = start_p
        self._trans_p = trans_p
        if prev_tags:
            self._prev_tags = prev_tags
        else:
            self._prev_tags = dict([(tag, self._tags) for tag in self._tags])
        if start_tags:
            self._start_tags = start_tags
        else:
            self._start_tags = self._tags
        if end_tags:
            self._end_tags = end_tags
        else:
            self._end_tags = self._tags

    def parse(self, node_scores):
        v = [{}]
        path = {}
        for t in self._tags:
            t_start_p = 0
            if self._start_p:
                t_start_p = self._start_p.get(t, MIN_FLOAT)
            v[0][t] = node_scores[0][t] + t_start_p
            path[t] = [t]
        for idx in range(1, len(node_scores)):
            v.append({})
            new_path = {}
            for tag in self._tags:
                (best_prob, best_tag) = max(
                    [(v[idx - 1][prev_tag] + self._trans_p[prev_tag].get(tag, MIN_FLOAT) + node_scores[idx][tag],
                      prev_tag) for prev_tag in self._tags if prev_tag[0] in self._prev_tags[tag[0]]])
                v[idx][tag] = best_prob
                new_path[tag] = path[best_tag] + [tag]
            path = new_path
        (best_prob, best_tag) = max(
            (v[len(node_scores) - 1][tag], tag) for tag in self._tags if tag[0] in self._end_tags)
        return best_prob, path[best_tag]
