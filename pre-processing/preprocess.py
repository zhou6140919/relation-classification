import json
import os
from itertools import permutations
import argparse


def preprocess(sent_file, tup_file, add_other=False):
    output_dir = 'datasets/processed_data'
    sents = open(sent_file).readlines()
    sents = [sent.strip() for sent in sents]

    tups = open(tup_file).readlines()
    tups = [tup.strip() for tup in tups]
    tups = [replace_relations(tup) for tup in tups]

    new_examples = []

    if not add_other:
        for s, t in zip(sents, tups):
            for _t in t.split(' | '):
                context = s
                e1, e2, rel = _t.split(' ; ')
                context = context + ' What is the relation between ' + e1 + ' and ' + e2 + '?'
                context = context.replace(e1, '<e1>' + e1 + '</e1>')
                context = context.replace(e2, '<e2>' + e2 + '</e2>')
                new_examples.append({'context': context, 'label': int(rel)})
    else:
        print("Adding other relations")
        for s, t in zip(sents, tups):
            entities = []
            label_dict = {}
            for _t in t.split(' | '):
                e1, e2, rel = _t.split(' ; ')
                entities.extend([e1, e2])
                label_dict[(e1, e2)] = int(rel)
            all_comb = list(permutations(entities, 2))
            for e1, e2 in all_comb:
                context = s
                if (e1, e2) in label_dict:
                    rel = label_dict[(e1, e2)]
                else:
                    rel = 29
                context = context + ' What is the relation between ' + e1 + ' and ' + e2 + '?'
                context = context.replace(e1, '<e1>' + e1 + '</e1>')
                context = context.replace(e2, '<e2>' + e2 + '</e2>')
                new_examples.append(
                    {'context': context, 'label': int(rel)})

    # data = [{'sent': s, 'tup': t} for s, t in zip(sents, tups)]
    print("length of data: ", len(new_examples))
    print(new_examples[2])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, os.path.basename(sent_file).replace('sent', 'json')), 'w') as w:
        for d in new_examples:
            w.write(json.dumps(d) + '\n')


def replace_relations(text):
    for rel in REL_DICT:
        text = text.replace(rel, str(REL_DICT[rel]))
    return text


def load_relation(rel_file):
    rels = open(rel_file).readlines()
    rels = [rel.strip() for rel in rels]
    rels_dict = {r: i for i, r in enumerate(rels)}
    return rels_dict


REL_DICT = load_relation('datasets/NYT29/relations.txt')
print(REL_DICT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_other', action='store_true')
    args = parser.parse_args()
    for mode in ['train', 'dev', 'test']:
        preprocess(f'datasets/NYT29/{mode}.sent',
                   f'datasets/NYT29/{mode}.tup', args.add_other)
