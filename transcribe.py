from pyannote.audio import Pipeline
from speechbrain.pretrained import EncoderASR
import pandas as pd
from pydub import AudioSegment
from pydeepspeech import transcribe
import subprocess

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1',
                                    use_auth_token="")
asr_model = \
EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-librispeech",
                        savedir="pretrained_models/asr-wav2vec2-librispeech",
                        run_opts={'device': 'cpu'})
own_file = 'interview1.wav'
OWN_FILE = {'audio': own_file}
diarization = pipeline(OWN_FILE)
with open('interview1.rttm', 'w') as rttm:
    diarization.write_rttm(rttm)
df = pd.DataFrame(columns=['Start_time', 'Stop_time', 'Speaker'])
for turn, _, speaker in diarization.itertracks(yield_label=True):
    df.loc[len(df.index)] = [turn.start, turn.end, speaker]
input_audio = AudioSegment.from_wav('./interview1.wav')
with open('interview1_transcript.txt', 'a') as f:
    for index, row in df.iterrows():
        if (round(row['Stop_time'] - row['Start_time'], 1) > 0.1):
            print(f"Processing for start time: {row['Start_time']} and end \
                    time: {row['Stop_time']} and for {row['Speaker']}")
            waveform = input_audio[(row['Start_time']*1000):
                                   (row['Stop_time']*1000)]
            waveform.export('temp.wav', format='wav')
            transcribed_text = asr_model.transcribe_file('temp.wav')
            f.write(row['Speaker'] + ':' + transcribed_text + '\n')
