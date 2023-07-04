from argparse import Namespace, ArgumentParser
import os
from time import sleep
import numpy as np

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
import whisper

stt = whisper.load_model("large", device="cpu") 

def inference(audio):
    
    buffer = whisper.pad_or_trim(audio[:,0])
    mel = whisper.log_mel_spectrogram(buffer).to(stt.device)

    _, probs = stt.detect_language(mel)
    
    lang=max(probs, key=probs.get)
    
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(stt, mel, options)
    
    print(result.text)
    return result.text



def main(args: Namespace) -> None:
  load_dotenv()
  os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
  
  if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)
    f = open(os.path.join(args.data_dir, "main.txt"), "w")
    f.write("SAMPLE DATA")
  
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
  
  duration = 8
  fs = 16000


  # Otherwise, we start a loop where we ask for user input.
  #with sr.Microphone() as source:
  while True:
    print("Listening...")
    
    #r.adjust_for_ambient_noise(source)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    
    
    #audio = r.listen(source, timeout=10)
    query = inference(audio)

    if not query:
      query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
      sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    
    wav = tts.tts(result["answer"], speaker=tts.speakers[1], language=tts.languages[0])

    sd.play(wav, samplerate=19000)
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