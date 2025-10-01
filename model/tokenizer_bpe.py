from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_stats(self, words):
        pairs = Counter()
        for w, freq in words.items():
            symbols = w.split(" ")
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, words):
        a, b = pair
        new_words = {}
        bigram = " ".join(pair)
        repl = a + b
        for w, freq in words.items():
            nw = w.replace(bigram, repl)
            new_words[nw] = freq
        return new_words

    def train(self, texts, vocab_size=12000, min_freq=2):
        words = Counter()
        for t in texts:
            for w in t.split():
                token = " ".join(list(w)) + "</w>"
                words[token] += 1
        words = Counter({w: c for w, c in words.items() if c >= min_freq})
        while len(self.vocab) < vocab_size:
            pairs = self._get_stats(words)
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            words = self._merge_vocab((a, b), words)
            self.merges.append((a, b))
            self.vocab["".join((a, b))] = freq
        tokens = set()
        for w in words.keys():
            tokens.update(w.split(" "))
        tokens.update(["<unk>", "<pad>"])
        self.token_to_id = {t: i for i, t in enumerate(sorted(tokens))}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def save(self, vocab_path, merges_path):
        import json

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False)
        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    @staticmethod
    def load(vocab_path, merges_path):
        import json

        tok = BPETokenizer()
        tok.token_to_id = json.load(open(vocab_path, "r", encoding="utf-8"))
        tok.id_to_token = {int(i): t for t, i in tok.token_to_id.items()}
        tok.merges = [
            tuple(line.strip().split())
            for line in open(merges_path, "r", encoding="utf-8")
            if line.strip()
        ]
        return tok

    def encode(self, text):
        def word_encode(w):
            symbols = list(w) + ["</w>"]
            merges = set(self.merges)
            changed = True
            while changed:
                changed = False
                i = 0
                while i < len(symbols) - 1:
                    pair = (symbols[i], symbols[i + 1])
                    if pair in merges:
                        symbols[i : i + 2] = ["".join(pair)]
                        changed = True
                    else:
                        i += 1
            return [
                self.token_to_id.get(s, self.token_to_id.get("<unk>")) for s in symbols
            ]

        ids = []
        for w in text.split():
            ids += word_encode(w)
        return ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        out = "".join(t.replace("</w>", " ") for t in tokens)
        return out.strip()
