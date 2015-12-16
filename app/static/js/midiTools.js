
var bpm = 240
var defaultNoteDuration = 1.0
var defaultNoteDuration = 1.0
var videoDuration = 6.5
var ANIMATION_DELTA_THRESHOLD = 0.15
// if change this, need to change the one over at resource.py also
var MIDI_START_SLACK = 0.15
var playbackStopTime = undefined
var DONT_INCLUDE_ONSET_OF_NEXT_NOTE_EPSILON = 0.05


var quartersPerSecond = bpm / 60.0
var secondToTickRatio = quartersPerSecond * sequencer.defaultPPQ
function secondToTicks(seconds) {
    return seconds * secondToTickRatio
}
function ticksToSecond(ticks) {
    return ticks / sequencer.defaultPPQ / quartersPerSecond
}

function makePlayEvents(notes, currentMidiTime, sequencer) {
// makes midi events from notes, and sends them off to be played right away
// returns the min onset as where the song is in time right now
    console.log('...makePlayEvents', currentMidiTime, notes.length)
    for (var i=0; i<notes.length; i++) {
        // start playing the first event right away
        var startTime = notes[i].onset - currentMidiTime
        var ticks = secondToTicks(startTime)
        console.log(i, notes[i].pitch, startTime, ticks)
        midiEvent = sequencer.createMidiEvent(ticks,
                sequencer.NOTE_ON, notes[i].pitch, velocity);

        sequencer.processEvents(midiEvent, bpm, instrumentName)
    }
}


function makeRandomNotes() {
    var randomNotes = []
    var numNotes = 10
    for (var i=0; i<numNotes; i++) {
        var note = {}
        note.onset = i/numNotes * videoDuration
        note.pitch = Math.floor((Math.random() * 40) + 1) + 40;
        randomNotes.push(note)
    }
    return randomNotes
}

function makeRandomSong(numNotes) {
    var part = sequencer.createPart()
    var events = sequencer.util.getRandomNotes({
            minNoteNumber: 60,
            maxNoteNumber: 100,
            minVelocity: 30,
            maxVelocity: 80,
            noteDuration: 120, //ticks
            numNotes: numNotes
    });
    part.addEvents(events)

    var track = sequencer.createTrack();
    track.setInstrument(instrumentName);
    track.addPart(part);

    var song = sequencer.createSong({
        bpm: bpm,
        tracks: track,
        useMetronome: false
    });
    return song
}


function makeNotes() {
    var randomNotes = []
    var pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    var numNotes = pitches.length
    var dur = videoDuration / numNotes
    for (var i=0; i<numNotes; i++) {
        var note = {}
        var onset = i/numNotes * videoDuration
        if (i==0) {
            note.onset = onset + ANIMATION_DELTA_THRESHOLD
        } else {
            note.onset = onset
        }
        note.pitch = pitches[i]
        note.offset = onset + dur
        randomNotes.push(note)
    }
    console.log('onsets', getOnsetList(randomNotes), randomNotes[randomNotes.length-1].offset)
    return randomNotes
}

function makeEvents(notes, sequencer) {
    var events = []
    var startTime, endTime, dur
    for (var i=0; i<notes.length; i++) {
        startTime = notes[i].onset // - minOnsetTime
        midiEvent = sequencer.createMidiEvent(secondToTicks(startTime),
                sequencer.NOTE_ON, notes[i].pitch, velocity);
        events.push(midiEvent)

        endTime = notes[i].offset
        midiEvent = sequencer.createMidiEvent(secondToTicks(endTime),
                sequencer.NOTE_OFF, notes[i].pitch, velocity);
        events.push(midiEvent)
        dur = endTime - startTime
//        console.log('makeEvents', i, startTime, endTime, dur)
    }
    // to avoid click, turning off the midi too quick
    // add a phantom note
    // a piano note has a 0.8s dur

    var padding = 0.9 - dur // 0.8 - dur
    if (padding > 0 && endTime+padding < videoDuration) {
//        console.log('...adding phantom note to end of phrase, padding', padding)
        midiEvent = sequencer.createMidiEvent(secondToTicks(endTime),
                sequencer.NOTE_ON, 60, 0);
        events.push(midiEvent)

        midiEvent = sequencer.createMidiEvent(secondToTicks(endTime+padding),
                    sequencer.NOTE_OFF, 60, 0);
        events.push(midiEvent)
//        console.log('makeEvents', 'phantom', endTime, endTime+padding)
    }
    return events
}

function makeSong(notes, sequencer) {
    console.log('notes', notes)
    var events = makeEvents(notes, sequencer)
    var part = sequencer.createPart()
    part.addEvents(events)

    var track = sequencer.createTrack();
    track.setInstrument(instrumentName);
    track.addPart(part);

    var song = sequencer.createSong({
        bpm: bpm,
        tracks: track,
        useMetronome: false
    });

    song.PPQ = sequencer.defaultPPQ
    console.log('inside song length', ticksToSecond(song.durationTicks))
    return song
}

function getMinOnsetTime(notes) {
    console.log('notes', notes)
    if (notes.length == 0) {
        return undefined
    }
    var minOnsetTime = notes[0].onset
    for (var i=1; i<notes.length; i++) {
        if (minOnsetTime > notes[i].onset) {
            minOnsetTime = notes[i].onset
        }
    }
    return minOnsetTime
}

function getOnsetList(notes) {
    var onsets = []
    for (var i=0; i<notes.length; i++) {
        onsets.push(notes[i].onset)
    }
    return onsets
}