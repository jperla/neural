#!/usr/bin/env python
import string

import numpy as np

import pyparse
from pyparse import Join, Any, Recursive, Keyword
from pyparse import AlphaWord, Nested, Repeat, ContiguousSymbols

# build up the tree parser
tree = Recursive()
root_tree = Recursive()
word = AlphaWord([p for p in string.printable if p not in '()'])
# POS tagset: http://aclweb.org/anthology-new/J/J93/J93-2004.pdf
pos_tokens = sorted([
                    'NN', 'PRP', 'VP', 'VB', 'VBP', 'NP', 
                    'DT', 'ADJP', 'ADVP', 'JJ', 'NNS',
                    'S', 'ROOT',
                    'PP', 'PRN',
                    'VBZ', 'VBG', 'VBN', 'VBD',
                    'CC', 'CD', 'DT', 'EX', 'FW',
                    'IN', 'JJR', 'JJS', 'LS', 'MD',
                    'NNP', 'NNPS', 'PDT', 'POS', 'PP$',
                    'RB', 'RBR', 'RBS', 'RP', 'SYM',
                    'TO', 'UH', 'WDT', 'WP$', 'WRB',
                    '.', '$', '#', ',', ':', '(', ')', '"', "'",
                    ], key=lambda s: -len(s))
pos = Any(*[Keyword(i) for i in pos_tokens])
subtree = Recursive()
#TODO: jperla: is Repeat right here?
# might be different for different POS?
# n:      (..subtree..)     
n = Join(None, Repeat(Keyword(' '), ignore=True), 
               Nested(content=subtree), 
               Repeat(Keyword(' '), ignore=True))
subtree_or_word = Any(word, Repeat(n))
subtree.update(None, Join(None, pos,    
                                Repeat(Keyword(' '), ignore=True),
                                subtree_or_word))
tree.update('TREE', n)
trees = Repeat(tree)


def read_stanford_parser(filename):
    """Accepts string filename.
        File has output of Stanford Parser.
        Returns AST of parse of file.
    """
    s = open(filename, 'r').read()
    w = pyparse.raw_tokenize(s)
    
    ast,remaining = trees.parse(w, whole=False)
    assert remaining == [], 'Could not parse whole file'
    return ast

class Node(object):
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right


def extract_parse_tree(stanford_parse):
    """Accepts an AST of a Stanford parse tree (with part of speech tags).
        Returns a new binary tree with only the words in a sentence.
        Right-associatively.

        Useful for computing word vectors on the phrases/sentences
    """
    def consolidate(l, r):
        """Accepts two lists of size 0-2.
            Returns new lists of size 0-2, 
                redistributing elements (greedily right), and
                removing empty lists.
        """
        not_none = [q for q in l + r if q != []]
        l, r = not_none[:-2], not_none[-2:]
        l = l[0] if len(l) == 1 else l
        r = r[0] if len(r) == 1 else r
        return l, r

    def glr(a):
        """Accepts a list of subtrees.
            Returns a 2-tuple of trees (basically for binary tree).
        """
        if len(a) == 0:
            return [], []
        elif len(a) == 1:
            #right = glr(a[0])
            if isinstance(a, tuple):
                return glr(a[0])
            else:
                return [], []
        elif len(a) == 2:
            if isinstance(a[1], tuple):
                if isinstance(a[0], tuple):
                    l,r = glr(a[0]), glr(a[1])
                    return consolidate(l, r)
                else:
                    return glr(a[1])
            else:
                r = a[1]
                #TODO: jperla: should I do this?
                # there are word vectors for periods/commas
                if r in string.punctuation:
                    return [], [] 
                else:
                    return [], r
        else:
            l,r = glr(a[:1]), glr(a[1:])
            return consolidate(l, r)
    left, right = glr(stanford_parse)
    assert left == []
    return right

def vectorize(vectors, params, tree):
    """Accepts the word embeddings, parameters for combining words,
        and a binary tree.
       Returns (recursively) a 4-tuple of 
            sentence length integer,
            full sentence string,
            embedding vector,
            and also returns a dictionary of "phrase" => vector.
                (for convenience)

            Or None if it's a null node.
    """
    if isinstance(tree, basestring):
        word = tree
        v = vectors[word]
        return 1, word, v, {word: v}

    if tree is None:
        return None

    left, right = tree[0], tree[1]
    if left is None:
        return vectorize(vectors, params, right)
    elif right is None:
        return vectorize(vectors, params, left)
    else:
        n1, phrase1, v1, phrases1 = vectorize(vectors, params, left)
        n2, phrase2, v2, phrases2 = vectorize(vectors, params, right)
        phrases = {}
        phrases.update(phrases1)
        phrases.update(phrases2)
        v = np.tanh((n1 * np.dot(params['W1'], v1)) +
                    (n2 * np.dot(params['W2'], v2)) + params['b1'])
        p = phrase1 + ' ' + phrase2
        phrases[p] = v
        return (n1 + n2), p, v, phrases

if __name__=='__main__':
    ast = read_stanford_parser('parsed.txt')
    trees = [extract_parse_tree(a) for a in ast]
    print trees

    import scipy.io
    v = scipy.io.loadmat('data/vars.normalized.100.mat')
    vocab_size = v['We'].shape[1]
    vectors = dict((v['words'][0,i][0], v['We'][:,i]) 
                                    for i in xrange(vocab_size))
    params = scipy.io.loadmat('data/params.mat')

    try:
        calculated = [vectorize(vectors, params, t) for t in trees]
    except Exception, e:
        print e
        import pdb; pdb.post_mortem()

    print calculated

    '''
    root = None
    visited = set([])
    current = ast
    stack = [current]
    while len(stack) == 0:
        assert len(current) > 1
        # leftmost is not tuple
        # middle ones are always tuples
        assert not isinstance(current[0], tuple)
        for i in xrange(1, len(current) - 1):
            assert isinstance(current[i], tuple)

        for i in xrange(1, len(current) - 1):
            if isinstance(current[-i], tuple):
                if current[-i] not in visited:
                    stack.append(current)
                    current = current[-i]
            else:
                root = Node(current[-1], None, root)
                
        visited.add(current)
        current = stack.pop()
    print root
    '''

