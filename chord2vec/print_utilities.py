

def print_seqs(seqs):
    num_display = 5
    if len(seqs) < num_display:
        num_display = len(seqs)
    num_display_chords = 15
    for seq in seqs[:num_display]:
        if len(seq) < num_display_chords:
            print seq
        else:
            print seq[:num_display_chords]
