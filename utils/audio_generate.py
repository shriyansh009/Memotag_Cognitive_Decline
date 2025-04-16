import pyttsx3
from pydub import AudioSegment

engine = pyttsx3.init()
engine.save_to_file("""Hello! Just a little reminder: You are safe and loved. 
If you’re unsure about anything or feel confused, it’s okay — help is nearby. 
Take a deep breath, maybe have a glass of water, and rest if you need to. 
Everything is taken care of. 
If you need anything, just ask — we’re here for you always.""", "sample2.wav")
engine.runAndWait()