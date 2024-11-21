import mido
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

N_USERS = 3
generated_notes = []
note_variances = []

for user_id in range(N_USERS):

    # path = f"C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/User Study/experiments/user_{user_id}/user_{user_id}_melody_generated.mid"
    path = f"C:/Users/Gianni/Desktop/MARCO/UNI/Magistrale/TESI/User Study/experiments/user_{user_id}/user_{user_id}_rhythm_played.mid"

    mid = mido.MidiFile(path)
    df = pd.DataFrame(columns=['note', 'time'])

    markers = []

    for i, track in enumerate(mid.tracks):
        time_sec = 0
        time_ticks = 0
        print(f"Track {i}: {track.name}")
        for msg in track:

            msg_time_sec = mido.tick2second(msg.time, mid.ticks_per_beat, mido.bpm2tempo(120))
            time_sec += msg_time_sec
            time_ticks += msg.time

            if msg.type == 'marker':
                minutes = int(time_sec // 60)
                seconds = int(time_sec % 60)
                print(f"Marker found: {msg.text} at {minutes}:{seconds}")
                markers.append({'text': msg.text, 'time': time_ticks})

            elif msg.type == 'note_on':
                df = pd.concat([df, pd.DataFrame({'note': [msg.note], 'time': [time_ticks]})], ignore_index=True)

    minutes = int(time_sec // 60)
    seconds = int(time_sec % 60)
    print(f'Last note played at: {minutes}:{seconds}')

    # # save dataframe to csv
    # df.to_csv('test.csv')

    # rows in df that are between two markers       
    relaxed_df = df[df['time'] < markers[1]['time']]
    excited_df = df[(df['time'] > markers[1]['time']) & (df['time'] < markers[2]['time'])]

    # count unique time values
    relaxed_notes = relaxed_df['time'].nunique()
    excited_notes = excited_df['time'].nunique()

    generated_notes.append(relaxed_notes)
    generated_notes.append(excited_notes)

    # calculate variance of notes played
    note_variances.append(relaxed_df['note'].var())
    note_variances.append(excited_df['note'].var())


# plot number of notes played
n = 2
relaxed = generated_notes[::n]
excited = generated_notes[1::n]
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(N_USERS)
bar1 = ax.bar(index, relaxed, bar_width, label='Relaxed', color='skyblue')
bar2 = ax.bar(index + bar_width, excited, bar_width, label='Excited', color='coral')

ax.set_xlabel('User')
ax.set_ylabel('Number of notes')
ax.set_title('Number of notes played in relaxed and excited state')
ax.set_xticks(index + bar_width / 2)
labels = []
for i in range(N_USERS):
    labels.append(f'User {i}')
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# save plot
if 'generate' in path:
    fig.savefig('generated_notes_per_class.png')
else:
    fig.savefig('played_notes_per_class.png')



# plot variance of notes
relaxed = note_variances[::n]
excited = note_variances[1::n]
fig, ax = plt.subplots()
bar1 = ax.bar(index, relaxed, bar_width, label='Relaxed', color='skyblue')
bar2 = ax.bar(index + bar_width, excited, bar_width, label='Excited', color='coral')

ax.set_xlabel('User')
ax.set_ylabel('Variance of notes')
ax.set_title('Variance of notes played in relaxed and excited state')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

# save plot
if 'generate' in path:
    fig.savefig('generated_variances_per_class.png')
else:
    fig.savefig('played_variances_per_class.png')