import pretty_midi


path = 'MIDI/examples/marker.mid'

pm = pretty_midi.PrettyMIDI(path)
print('Number of instruments:', len(pm.instruments))

instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
print('Instrument name:', instrument_name)


print(f'Time signature: {pm.time_signature_changes}')
print(f'Tempo changes: {pm.get_tempo_changes()}')
print(f'Key signature: {pm.key_signature_changes}')
print(f'Lyrics: {pm.lyrics}')
print(f'Text Events: {pm.text_events}')


for i, note in enumerate(instrument.notes):
  note_name = pretty_midi.note_number_to_name(note.pitch)
  duration = note.end - note.start
  print(f'{i}: pitch={note.pitch}, note_name={note_name},'f' duration={duration:.4f}')