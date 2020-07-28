import numpy as np
import json
import re
from PIL import Image, ImageDraw
import os
import itertools

nlp = None


def states_to_label(state0, state1):
    for p0, p1 in zip(state0, state1):
        if p0 != p1:
            xs, ys = p0[0], p0[2]
            xf, yf = p1[0], p1[2]
            return (xs, ys), (xf, yf)


def states_to_idx(state0, state1):
    for idx, (p0, p1) in enumerate(zip(state0, state1)):
        if p0 != p1:
            return idx


def state_to_img(state):
    w, h = 150, 150
    img = Image.new('RGB', (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    sz = 8
    state_colors = list(itertools.product([0, 127, 255], [0, 127, 255], [0, 127, 255]))[:-1]
    for idx, (x, _, y) in enumerate(state):
        px = int((x + 1) * 0.5 * w)
        py = int((-y + 1) * 0.5 * h)
        draw.rectangle([(px - sz // 2, py - sz // 2), (px + sz // 2, py + sz // 2)], fill=state_colors[idx])

    return np.array(img)


SENT_N_TOKENS = 40
VOCAB_SIZE = 806


def read_vocab(vocab_fn='vocab_digit'):
    v = open(vocab_fn).read().split('\n')
    v = [w for w in v if len(w) > 0]
    v = dict([(w.lower(), idx) for idx, w in enumerate(v)])
    return v


vocabs = {}


def sent_to_idxs(sent, vocab_fn='vocab_digit'):
    global vocabs
    if vocab_fn not in vocabs:
        v = open(vocab_fn).read().split('\n')
        v = [w for w in v if len(w) > 0]
        print(f'Vocab size of "{vocab_fn}": {len(v)}')
        v = dict([(w.lower(), idx) for idx, w in enumerate(v)])
        vocabs[vocab_fn] = v

    vocab = vocabs[vocab_fn]

    # sent = sent.replace('.', ' ')
    # sent = sent.replace(',', ' ')
    # sent = sent.replace(';', ' ')
    # sent = sent.replace(':', ' ')
    # sent = sent.replace("'", ' ')
    sent = ''.join(c if c.isalnum() else ' ' for c in sent)
    sent = re.sub(r'([0-9])th ', r'\1 th ', sent)
    # if re.search(r'[0-9] th ', sent): print(sent)

    idxs = []
    for word in sent.split(' '):
        if len(word) > 0:
            word = word.lower()
            if word in vocab:
                idxs.append(vocab[word])

    if len(idxs) > SENT_N_TOKENS: return None

    idxs = idxs[:SENT_N_TOKENS]
    if len(idxs) < SENT_N_TOKENS:
        idxs += [0] * (SENT_N_TOKENS - len(idxs))

    return idxs


def make_dataset(fn='trainset.json', vocab_fn='vocab_digit'):
    Ss, Ts, Xs, As, Ys, Yis = [], [], [], [], [], []

    data = json.load(open(fn))

    for dat in data:
        if dat['decoration'] == 'digit':
            states = dat['states']
            np_img_states = [state_to_img(state) for state in states]

            for d in dat['notes']:
                sents = d['notes']
                start = d['start']
                finish = d['finish']
                typ = d['type']

                if typ != 'A0': continue

                if len(states[start]) != 20: continue

                (xs, ys), (xf, yf) = states_to_label(states[start], states[finish])
                state_idx = states_to_idx(states[start], states[finish])
                np_state_pos = np.array(
                    [(x, y) for (x, _, y) in states[start]] + [(0, 0)] * (21 - len(states[start])) + [(1, 1)])

                for sent in sents:
                    sent_idxs = sent_to_idxs(sent, vocab_fn=vocab_fn)
                    if not sent_idxs: continue
                    Ss.append(sent.lower())
                    Ts.append(sent_idxs)
                    As.append(np_state_pos)
                    Xs.append(np_img_states[start])
                    Ys.append(np.array([xs, ys, xf, yf]))
                    Yis.append(state_idx)

    Ss = np.array(Ss, dtype=object)[:, None]
    Ts = np.array(Ts)
    Xs = np.array(Xs)
    As = np.array(As)
    Ys = np.array(Ys)
    Yis = np.array(Yis)

    return Ss, Ts, Xs, As, Ys, Yis


def make_train_dataset():
    return make_dataset(fn='trainset.json', vocab_fn='vocab_digit')


def make_dev_dataset():
    return make_dataset(fn='devset.json', vocab_fn='vocab_digit')


def make_test_dataset():
    return make_dataset(fn='testset.json', vocab_fn='vocab_digit')


def main():
    Ss, Ts, Vs, Xs, As, Ys, Yis = make_dataset()

    print(sorted([len(Ts[i]) for i in range(len(Ts))])[-1000], Ss.shape, Ts.shape, Vs.shape, Xs.shape, As.shape,
          Ys.shape, Yis.shape)

    os.system('rm -f tmp/*.png')

    idx = 0
    Image.fromarray(Xs[idx, :, :, :]).save(f'tmp/x_{idx:03d}_{Ys[idx][0]}_{Ys[idx][1]}_{Ys[idx][2]}_{Ys[idx][3]}.png')

    idx = 9
    Image.fromarray(Xs[idx, :, :, :]).save(f'tmp/x_{idx:03d}_{Ys[idx][0]}_{Ys[idx][1]}_{Ys[idx][2]}_{Ys[idx][3]}.png')


if __name__ == '__main__':
    main()