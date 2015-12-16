    
// load midi sounds
window.onload = function () {
	MIDI.loadPlugin({
		soundfontUrl: "./soundfont/",
		instrument: ["acoustic_grand_piano", "acoustic_guitar_nylon", "acoustic_guitar_steel", "celesta"],
		callback: function() {
            MIDI.programChange(0, 0);
			MIDI.programChange(1, 24);
			MIDI.programChange(2, 25); 
            MIDI.programChange(3, 8);            
			playMidiDebounced([60])
		}
	});
};

var velocity = 100;
var dur = 0.45; 
var allNotes = []
var channel = 0

function playMidiSeq(noteSeqs, durs) {   
    var delay = 0
    for (var i=0; i<noteSeqs.length; i++) {
        notes = noteSeqs[i]
        for (var j=0; j<notes.length; j++) {
            MIDI.setVolume(channel, velocity);
            MIDI.noteOn(channel, notes[j], velocity, delay);
            MIDI.noteOff(channel, notes[j], delay + durs[i]+0.05);
            allNotes.push([notes[j], delay])
        }
        delay += durs[i]
    }
}

// play midi
function playMidi(notes) {
    for (var i=0; i<notes.length; i++) {
        var delay = 0; // play one note every quarter second
        // play the note
        MIDI.setVolume(channel, velocity);
        MIDI.noteOn(channel, notes[i], velocity, delay);
        MIDI.noteOff(channel, notes[i], delay + dur);
        allNotes.push([notes[i], delay])
    };
};

playMidiDebounced = _.debounce(function(notes) { playMidi(notes) }, 300);
playMidiSeqDebounced = _.debounce(function(noteSeqs, durs) { playMidiSeq(noteSeqs, durs) }, 300);
