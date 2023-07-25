
import openai
from dotenv import load_dotenv
import os
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

load_dotenv()

# Sampling frequency

freq = 44100

# Recording duration

duration = 25

if "OPENAI_KEY" in os.environ:
    openai.api_key = os.getenv("OPENAI_KEY")


def check_intent(transcript):
    response = openai.ChatCompletion.create(

        model='gpt-3.5-turbo',

        messages=[

            {

                "role": "system",

                "content": f"""You have been given the following:{transcript}

                                This is an audio transcript of a song. You are a highly creative AI-powered songwriter. 
                                Your task is to transform the provided paragraph into a suitable structure for song lyrics. 
                                The paragraph may contain emotions, stories, or ideas that should be beautifully woven into engaging and poetic lyrics. 
                                Your goal is to craft a set of lyrics that capture the essence of the paragraph and evoke a powerful emotional response when sung. 
                                Let your imagination flow and create a mesmerizing song that speaks to the heart!
                                

                                Give the output in the following format:

                                Give gap of a line wherever necessary

                            """

            }

        ], temperature=0.2, max_tokens=749, top_p=1, frequency_penalty=0, presence_penalty=0)

    return response['choices'][0]['message']['content']


st.header('Lyrics Writer')
# Create a text input widget







uploaded_file = st.file_uploader("Choose an audio file", type=['.wav', '.wave', '.flac', '.mp3', '.ogg'],
                                 accept_multiple_files=False)

initiate = st.button("Recorder")

button = st.button("Submit")

if initiate:
    recording = sd.rec(int(duration * freq),

                       samplerate=freq, channels=1)

    # Record audio for the given number of seconds

    sd.wait()

    with st.spinner('Wait for it...'):
        wv.write("recording1.wav", recording, freq, sampwidth=2)

        audio_file = open("recording1.wav", "rb")

        transcript = openai.Audio.transcribe(

            file=audio_file,

            model="whisper-1",

            response_format="text",

            language="en"

        )

    st.write(transcript)

    intent = check_intent(transcript)

    st.success(intent)

if button:
    transcript = openai.Audio.transcribe(

        file=uploaded_file,

        model="whisper-1",

        response_format="text",

        language="en"

    )

    st.write(transcript)

    intent = check_intent(transcript)

    st.success(intent)


