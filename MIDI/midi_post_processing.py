import mido


path = "C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/User Study/experiments/user_2/user_2_melody_generated.mid"

mid = mido.MidiFile(path)

markers = []
notes = []

for i, track in enumerate(mid.tracks):
    time = 0
    print(f"Track {i}: {track.name}")
    for msg in track:
        if msg.type == 'marker':
            time += msg.time 
            print(f"Marker found: {msg.text} at index {time}")
            markers.append({'text': msg.text, 'time': time})

        elif msg.type == 'note_on':
            time += msg.time
            notes.append({'note': msg.note, 'time': time})

print(markers)
print(len(notes))
            
