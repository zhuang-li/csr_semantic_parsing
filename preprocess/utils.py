import spacy
def lemma(dataset):
    sp = spacy.load('en')
    lem = []
    for data in dataset:
        sent = sp(data)
        temp = []
        for word in sent:
            if word.lemma_ != '-PRON-':
                temp.append(word.lemma_)
            else:
                temp.append(word.lower_)
        lem.append(' '.join(temp))
    return lem

