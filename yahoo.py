from lxml import etree
import xml
import gensim
import collections as coll
import numpy as np
import nltk

def load_data(path='manner.xml',ratio = 5,exp = 'all', nb_words=None, skip_top=0, maxlen=None, test_split=0.2,seed=113, start_char=1, oov_char=2, index_from=3):
    """
    returns 2 tuples: (X_train,X_test),(y_train,y_test) as a sequence of embeddings
    """
    def transform(model,s):
        #check to make sure gensim/word2vec deals with missing words correctly
        example = []
        for token in s.split():
            if token == ' <S> ':
                example.append(np.ones(300))
            if token in model:
                example.append(model[token])
        return example
    def vocab(v,q_a):
        sequence = []
        for s in q_a.split():
            if s not in v:
                v[s] = len(v)
            sequence.append(v[s])
        return sequence

    X = []
    y = []
    raw = get_data(path)
    # model = gensim.models.Word2Vec.load_word2vec_format('w2v/GoogleNews-vectors-negative300.bin.gz',binary=True)
    model = {}
    #each class
    for k in raw.iterkeys():
        #each question
        for q_a in raw[k]:
            correct = vocab(model,' <S> '.join(q_a))
            X.append(correct)
            y.append(1)
            for i in xrange(ratio):
            #select only from subcat

                X.append(vocab(model,q_a[0]+' <S> ' +
                                   raw[k][np.random.randint(len(raw[k]))][1]))
                y.append(0)
    split = int(len(X)*(1-test_split))
    print "train: ",len(X[split:])
    print "test:  ",len(X[:split])
    return (X[:split],y[:split]),(X[split:],y[split:])





def get_data(path='manner.xml'):
    """
    returns dict of type:(question,answer)
    """
    def extract(elem_q,elem_a,elem_mcat):
        ans = elem_a.text
        q = elem_q.text

        punctCodes = (
                ('', '?'),
                (' ', '\'s'),
                ('', '!'),
                (' ', ','),
                (' ', ';'),
                (' ','.'),
                (' ','('),
                (' ',')'),
                (' ','['),
                (' ',']'),
                ('$ ','$'),
                (' ','-'),
                ('', '"'),
                ('', 'br /'),
                (" ", '\n'),
                ("", '&#39;'),
                ('', '&quot;'),
                ('', '&gt;'),
                ('', '&lt;'),
                ('','<>'),
                (' and ', '&amp;')
            )
        for code in punctCodes:
            ans = ans.replace(code[1], code[0])
            q = q.replace(code[1], code[0])
        return (q,ans)


    def fast_iter(context, func):
        stats=  coll.defaultdict(int)
        q_type = coll.defaultdict(list)

        i = 0
        for event, elem in context:
            # if elem.tag == 'uri':
            #     elem_uid = elem
            if elem.tag == 'subject':
                elem_q = elem
            if elem.tag == 'bestanswer':
                elem_a = elem
            if elem.tag == 'nbestanswers':
                elem_na = elem
            if elem.tag == 'maincat':
                elem_mcat = elem
                stats[elem_mcat.text] += 1
                if (i+1)%500 == 0:
                    break
                    print '. ',

                    #TODO move outside of if statement once done testing
                result = func(elem_q,elem_a,elem_mcat)
                q_type[elem_mcat.text].append(result)
                i+=1
                elem.clear
                for ancestor in elem.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]
        del context
        print "processed ",i+1," questions"
        return q_type

    context = etree.iterparse(path)
    func = extract
    sent_detector = 'f'
    result = fast_iter(context,func)
    #expect list of question:best-answer
    return result


if __name__ == '__main__':
    d = {}
    t = 'the fast dog wen\'t home the'
    load_data()
