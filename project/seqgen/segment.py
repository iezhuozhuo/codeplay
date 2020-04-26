# -*- coding: UTF-8 -*-
import sys
import re
from collections import Counter
import multiprocessing as mp
import os
import argparse

rNUM = re.compile(u'(-|\+)?\d+((\.|·)\d+)?%?')
rENG = re.compile(u'[A-Za-z_.]+')

hashtags = [u'##', u'()', u'**', u'[]', u'（）', u'【】', u'\"\"', u'“”', u'(）', u'（)', u'【】']
patterns = [re.compile(u"\%s.*?\%s" % (ht[0], ht[1])) for ht in hashtags]


def remove_hashtag(sent):
    new_sent = sent
    for pattern in patterns:
        subs = pattern.findall(sent)
        if len(subs) > 0:
            for sub in subs:
                new_sent = new_sent.replace(sub, '')

    new_sent = new_sent.replace("  ", ' ')
    return new_sent.strip()


def segment_line(line, v, cnt):
    words = remove_hashtag(line.strip()).split()
    if v is not None:
        new_words = []
        for w in words:
            if w in v:
                new_words.append(w)
            else:
                if rNUM.match(w) is not None or rENG.match(w) is not None:
                    new_words.append(w)
                else:
                    for c in list(w):
                        new_words.append(c)
        words = new_words

    new_words = [None]
    for w in words:
        if w!= '<BLANK>' and re.sub('\W','',w,flags=re.U) != w:
            if (w == new_words[-1]):
                continue
            else:
                new_words.append(w[0])
        else:
            new_words.append(w)
    new_words = new_words[1:]
    last_word = ""
    while (len(new_words)>0 and re.sub('\W','',new_words[-1], flags=re.U)!=new_words[-1]):
        last_word = new_words[-1]
        new_words = new_words[:-1]
    if last_word != "":
        new_words.append(last_word)
    cnt.update(new_words)
    return ' '.join(new_words)


def segment(lines, v):
    cnt = Counter()
    new_lines = []
    for line in lines:
        new_line = [segment_line(x, v, cnt) for x in line.strip().split('|')]
        new_lines.append('|'.join(new_line))
    return new_lines, cnt


def fast_segment(lines, v, workers, lines_per_work):
    def merge_result(result):
        lines_i, cnt_i = result
        new_lines.extend(lines_i)
        cnts.append(cnt_i)

    pool = mp.Pool(workers)
    new_lines = []
    cnts = []

    N = len(lines)
    start_id = 0
    while start_id < N:
        cur = lines[start_id:start_id + lines_per_work]
        pool.apply_async(segment, args=(cur, v), callback=merge_result)
        start_id += lines_per_work
    pool.close()
    pool.join()
    cnt = cnts[0]
    for x in cnts[1:]:
        cnt += x
    return new_lines, cnt


def segment_file(fname, vocab_file, nworkers, lines_per_work):
    lines = [x for x in open(fname, encoding="utf-8", errors='ignore').readlines()]
    v = set([x.strip() for x in open(vocab_file).readlines()])
    v.add('<BLANK>')
    lines, _ = fast_segment(lines, v, nworkers, lines_per_work)

    with open(fname +'_processed', 'w') as fo:
        for line in lines:
            ok = True
            for x in line.strip().split('|'):
                if len(x) ==0:
                    ok = False
            if ok:
                fo.write(line.encode('utf8')+'\n')


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--train_file', type=str, default="train.txt")
    parser.add_argument('--output_dir', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    where = os.path.join(args.train_dir, args.train_file)
    nworkers = 8
    lines_per_work = 200000
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #segment_file(where, '../bigger_golden/vocab_src', nworkers, lines_per_work)
    #exit(0)
    lines = [x for x in open(where, encoding="utf-8", errors='ignore').readlines()]
    lines, cnt = fast_segment(lines, None, nworkers, lines_per_work)

    print('first_round done')

    vocab = set(x[0] for x in cnt.most_common()[:20000])
    lines, cnt = fast_segment(lines, vocab, nworkers, lines_per_work)

    print('second_round done')
    vocab = set(x[0] for x in cnt.most_common()[:20000])
    lines, cnt = fast_segment(lines, vocab, nworkers, lines_per_work)

    print('finishing')
    vocab_save_path = os.path.join(args.output_dir, 'vocab')
    with open(vocab_save_path, 'w', encoding="utf-8", errors='ignore') as fo:
        for w in vocab:
            fo.write(w + '\n')

    processed_save_file = os.path.join(args.output_dir, 'train_processed')
    with open(processed_save_file, 'w', encoding="utf-8", errors='ignore') as fo:
        for line in lines:
            ok = True
            for x in line.strip().split('|'):
                if len(x) ==0:
                    ok = False
            if ok:
                fo.write(line +'\n')
