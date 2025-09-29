import streamlit as st
import gdown
import tensowflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def carrega_modelo():
    # Corrigi o link do Google Drive (era "if=1..." mas o correto é "id=")
    url = 'https://drive.google.com/uc?id=1yPP8wwVLbmrQjw0qLfs3DBHVR035tmzH'

    gdown.download(url, 'modelo_vidente.keras')
    loaded_model = tf.keras.models.load_model('modelo_vidente.keras')
    with open('vectorizer.pkl','rb') as file:
      vectorizer = pickle.load(file)
    return loaded_model, vectorizer

def predict_next_words(modelo, vectorizer, text, max_sequence_len, top_k=3):
    tokenized_text = vectorizer([text]) #tokenização dos textos
    tokenized_text = np.squeeze(tokenized_text) #remoção de dimensões
    padded_text = pad_sequences([tokenized_text], maxlen=max_sequence_len, padding='pre') #preenchimento por 0s
    predicted_probs = modelo.predict(padded_text, verbose=0)[0] #probabilidade de ser a próxima palavra
    top_k_indices = np.argsort(predicted_probs)[-top_k:][::-1] #retorna o índice das palavras
    predicted_words = [vectorizer.get_vocabulary()[index] for index in top_k_indices]
    return predicted_words


def main():

  max_sequence_len = 50

  loaded_model, vectorizer = carrega_modelo()
  
  st.title('Previsão de Próximas Palavras')
  input_text = st.text_input('Digite uma sequencia de texto:')

  if st.button('Prever'):
    if input_text:
      try:
        predicted_words = predict_next_words(loaded_model, vectorizer, input_text, max_sequence_len)

        st.info('Palavras mais provaveis')
        
        for word in predicted_words:
          st.success(word)
      except Exception as e:
        st.error('Deu erro ze {e}')

    else:
      st.warning('Insira algo')

if __name__ == "__main__":
  main()
