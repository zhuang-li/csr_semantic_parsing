# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# Press the green button in the gutter to run the script.
from preprocess.event_extractor import event_match_embedding,utterance2embedding,event2embedding
import os.path
from os import path

if __name__ == '__main__':

    utter_path='./data/atomic_data/userUtterSentences.csv'

    event_embedding_path= 'data/atomic_data/event_embeddings_base.npy'

    utterance_embedding_path= 'data/atomic_data/utterance_embeddings_base.npy'
    result_path = './data/atomic_data/utter_event.csv'
    if not path.exists(event_embedding_path):
        event2embedding(event_embedding_path)
    if not path.exists(utterance_embedding_path):
        utterance2embedding(utterance_embedding_path, utter_path)

    event_match_embedding(utter_path, event_embedding_path,utterance_embedding_path, result_path=result_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
