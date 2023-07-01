from argparse import Namespace, ArgumentParser
import os
from time import sleep

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import sys
import speech_recognition as sr
import sounddevice as sd
from TTS.api import TTS

def main(args: Namespace) -> None:
  load_dotenv()
  os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
  
  PERSIST = False
  
  if PERSIST and os.path.exists(args.index_dir):
    print("\nReusing index...\n")
    vectorstore = Chroma(persist_directory=args.index_dir, embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
  else:
    loader = DirectoryLoader(args.data_dir)
    if PERSIST:
      index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": args.index_dir}).from_loaders([loader])
    else:
      index = VectorstoreIndexCreator().from_loaders([loader])
      
      
  chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model=args.model),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
  )
  
  chat_history = []
  
  # If an user input is provided, we just use that and exit.
  if args.user_input:
    result = chain({"question": args.user_input, "chat_history": chat_history})
    print(result['answer'])
    exit()

  query = None
  
  r = sr.Recognizer()
  model_name = TTS.list_models()[0]
  tts = TTS(model_name)
  

  # Otherwise, we start a loop where we ask for user input.
  with sr.Microphone() as source:
    while True:
      print("Listening...")
      
      r.adjust_for_ambient_noise(source)
      
      audio = r.listen(source, timeout=10)
      
      # Perform speech recognition
      try:
          text = r.recognize_google(audio)
          print("Recognized text:", text)
          query = text
      except sr.UnknownValueError:
          print("Speech recognition could not understand audio")
      except sr.RequestError as e:
          print("Could not request results from Google Speech Recognition service; {0}".format(e))
      
      
      if not query:
        query = input("Prompt: ")
      if query in ['quit', 'q', 'exit']:
        sys.exit()
      result = chain({"question": query, "chat_history": chat_history})
      
      wav = tts.tts(result["answer"], speaker=tts.speakers[0], language=tts.languages[0])

      sd.play(wav, samplerate=23000)
      sd.wait()  

      chat_history.append((query, result['answer']))
      query = None


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("user_input", type=str, nargs="?", default=None)
  parser.add_argument("--data_dir", type=str, default="./data")
  parser.add_argument("--index_dir", type=str, default="./index")
  parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
  args = parser.parse_args()
  
  main(args)