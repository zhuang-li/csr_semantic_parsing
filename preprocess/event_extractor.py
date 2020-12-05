import re
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from nltk import tokenize
def read_atomic_data(file_path):
    atomic_df = pd.read_csv(file_path, encoding='utf8')
    atomic = []
    atomic_org = []
    for row in range(len(atomic_df)):
        temp = re.sub('___', '_', atomic_df['event'][row])
        atomic_org.append(''.join([temp]))
        temp = re.sub('(?i)personx', '', temp)
        temp = re.sub('(?i)persony', '', temp)
        temp = re.sub('(?i)personz', '', temp)
        atomic.append(''.join([temp]))


def event_match_embedding(utter_path, event_embedding_path, utterance_embedding_path, result_path, choice_num = 4):

    #model = SentenceTransformer('./roberta-base-nli-stsb-mean-tokens')
    #model = SentenceTransformer("/home/taof/da33/tao/sentence-transformer-models/roberta-large-nli-stsb-mean-tokens")

    with open(utter_path, "r") as f:
        utterances = f.readlines()
        for i in range(len(utterances)):
            utterances[i] = utterances[i].replace("_comma_", ",")

    newUtterances = []
    for utterance in utterances:
        newUtterances.extend(tokenize.sent_tokenize(utterance))

    # load events embeddings
    eventEmbeddings = np.load(event_embedding_path)
    eventNum, dim = eventEmbeddings.shape
    # build the index
    # index = faiss.IndexFlatL2(dim)
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(eventEmbeddings)  # cosine similarity only
    index.add(eventEmbeddings)

    utteranceEmbeddings = np.load(utterance_embedding_path)
    print (utteranceEmbeddings.shape)
    utterance2event = []
    for u_id, utterance in enumerate(tqdm(newUtterances)):
        # minDistance = 100000
        cosineSimilar = -10
        eventIndex = ""
        queryEmbedding = utteranceEmbeddings[u_id].reshape(1,-1)
        #print(queryEmbedding.shape)
        faiss.normalize_L2(queryEmbedding)  # cosine similarity only
        D, I = index.search(queryEmbedding, choice_num)
        # if D[0][0] < minDistance:
        event_indx_list = []
        for event_idx in range(choice_num):
            if D[0][event_idx] > cosineSimilar:
                # minDistance = D[0][0]
                #cosineSimilar = D[0][0]
                eventIndex = I[0][event_idx]
                event_indx_list.append(eventIndex)
        utterance2event.append(event_indx_list)

    with open("./data/atomic_data/all-events.csv", "r") as f:
        allEvents = f.readlines()

    with open(result_path,'w') as f:
        for u_id, utter in enumerate(newUtterances):
            f.write(utter + '\n')
            for event_indx in utterance2event[u_id]:
                f.write(allEvents[event_indx])
            f.write('\n')
        f.close()


def utterance2embedding(dump_path, utter_path):
    model = SentenceTransformer('./roberta-base-nli-stsb-mean-tokens')
    #model = SentenceTransformer("/home/taof/da33/tao/sentence-transformer-models/roberta-large-nli-stsb-mean-tokens")

    with open(utter_path, "r") as f:
        utterances = f.readlines()
        for i in range(len(utterances)):
            utterances[i] = utterances[i].replace("_comma_", ",")
    f.close()
    newUtterances = []
    for utterance in utterances:
        newUtterances.extend(tokenize.sent_tokenize(utterance))

    utterEmbeddings = model.encode([newUtterances[0]])
    for new_utter in tqdm(newUtterances[1:]):
        utter_Embedding = model.encode([new_utter])
        utterEmbeddings = np.append(utterEmbeddings, utter_Embedding, axis=0)

    with open(dump_path, "wb") as f:
        np.save(f, utterEmbeddings)

def event2embedding(dump_path):
    model = SentenceTransformer("./roberta-base-nli-stsb-mean-tokens")

    fr = open("./data/atomic_data/all-events.csv", "r")
    events = fr.readlines()
    fr.close()

    eventEmbeddings = model.encode([events[0]])
    for event in tqdm(events[1:]):
        newEmbedding = model.encode([event])
        eventEmbeddings = np.append(eventEmbeddings, newEmbedding, axis=0)

    with open(dump_path, "wb") as f:
        np.save(f, eventEmbeddings)