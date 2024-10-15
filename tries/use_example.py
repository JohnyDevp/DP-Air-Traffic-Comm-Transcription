import whisper

model = whisper.load_model("base")

result = model.transcribe("record_cz.flac")
# model.transcribe()
# result = model.detect_language("record_cz.flac")
print(result)
