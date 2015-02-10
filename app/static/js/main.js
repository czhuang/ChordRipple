
function status(x) {
  console.log('status', x);
  $('#status').text(x);
}

var withoutPicker = true

var socket = undefined; 
var inputHistory = [];
var vertices = [];
var labels = [];

// d3 picker interface...
var w = 750, 
    h = 750;

$(document).ready(function() {

  // connecting
  status('Connecting...');
  WEB_SOCKET_DEBUG = true;
  socket = io.connect("");

  socket.on('connect', function() {
    status('Connected.');
    socket.emit('ping', "Hello World!");
    //socket.emit("requestData", function(){ console.log("...emitted data request") });
    //socket.emit("requestStarters", function() { console.log("...emitted starters request") });
  });
  socket.on('reconnect', function () {
    status('Reconnected.');
  });
  socket.on('reconnecting', function (msec) {
    status('reconnecting in '+(msec/1000)+'sec ... ');
    $("#status").append($('<a href="#">').text("Try now").click(function(evt) {
      evt.preventDefault();
      socket.socket.reconnect();
    }));
  });
  socket.on('connect_failed', function() { status('Connect failed.'); });
  socket.on('reconnect_failed', function() { status('Reconnect failed.'); });
  socket.on('error', function (e) { status('Error: '+ e); });

  socket.on('pong', function(arg) {
    console.log("Pong! "+arg);
    //status("Pong! "+arg);
  });
  
  
  // playback
  socket.on("playNotes", function(midiNotes) {
    console.log('playNotes', midiNotes)
    playMidiDebounced(midiNotes)
  });
  
  socket.on("playSeq", function(noteSeqs, durs) {
    console.log('playSeq', noteSeqs, durs)
    playMidiSeqDebounced(noteSeqs, durs)
  });
  
  socket.on("updateChordSuggestions", function(subChords, subInds, inputChordSeq) {
    updateSuggestions(subChords, subInds, inputChordSeq) 
  });
  
  socket.on("updateChordSuggestionsSimple", function(chordSeqs, startInd) {
    updateSuggestionsSimple(chordSeqs, startInd)
  });

  
  socket.on("updateHistory", function(previousInput, currentInput) {
    // update history display
    if (previousInput != currentInput) {
        inputHistory.push(currentInput)
        diffStrs(previousInput, currentInput)
    };
  });

  socket.on("dataLabels", function(receivedData, receivedTextLabels) {
    vertices = receivedData
    textLabels = receivedTextLabels
    console.log('...received', vertices, textLabels)
    if (!withoutPicker){
        setupPicker(vertices, textLabels, socket)
    }
  });
});


// load midi sounds
window.onload = function () {
	MIDI.loadPlugin({
		soundfontUrl: "./soundfont/",
		instrument: "acoustic_grand_piano",
		callback: function() {
			playMidiDebounced([60])
		}
	});
};

var velocity = 50;
var dur = 0.75; 
var allNotes = []

function playMidiSeq(noteSeqs, durs) {
    var delay = 0
    for (var i=0; i<noteSeqs.length; i++) {
        notes = noteSeqs[i]
        for (var j=0; j<notes.length; j++) {
            MIDI.setVolume(0, velocity);
            MIDI.noteOn(0, notes[j], velocity, delay);
            MIDI.noteOff(0, notes[j], delay + durs[i]);
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
        MIDI.setVolume(0, velocity);
        MIDI.noteOn(0, notes[i], velocity, delay);
        MIDI.noteOff(0, notes[i], delay + dur);
        allNotes.push([notes[i], delay])
    };
};

playMidiDebounced = _.debounce(function(notes) { playMidi(notes) }, 300);
playMidiSeqDebounced = _.debounce(function(noteSeqs, durs) { playMidiSeq(noteSeqs, durs) }, 300);

// setup UI
var widthPercent = '85%' 
parent = document.getElementById('chart');
var headerText = 'ChordRipple '
var subheaderText = "Change one chord, and see the changes propagate down the list.  See substitutions, what's next, surrounding chords change, and continuations that lead you to the end of a phrase. " 
var header = $('<h1>').text(headerText).appendTo(parent)
var subheader = $('<small>').text(subheaderText).appendTo(header)

var inputTextInstructText = 'Type chords here as roman numerals (i.e. I ii IV).  Letter chords to come soon, and better voicing of chords.  Right now chords are played as a block. '
var inputTextInstruct = $('<h5>').text(inputTextInstructText).appendTo(parent)

inputText = $('<input>').addClass('text').attr('id', 'inputtext').css({"font-family":"monospace", "font-size":16, "margin-top":8}).width(widthPercent).appendTo(parent);
inputText.focus(function(e) {
    if (inputText.val().length == 0) {
        socket.emit("startSeqs")
    }
});

inputTextPlayButton = $('<button>').addClass('btn').text('Play!').appendTo(parent);
inputTextPlayButton.click(function(e) {
    socket.emit("playSeq", inputText.val(), inputText.caret().start)
});

var suggestList = []
var suggestTexts = []
var suggestInd = -1
var maxNumSuggestions = 15
var desiredLen = 50

createSuggestions(parent)

function createSuggestions(parent) {
    for (var i=0; i<maxNumSuggestions; i++) {
        var p = $('<p>').attr('id', 'suggest'+i).appendTo(parent)
        suggestTexts.push('')
    }
}

function updateSuggestions(subChords, subInds) {
    for (var i=0; i<15; i++) {
        parent = $('#suggest'+i.toString())
        parent.empty()
    }

    for (var i=0; i<subChords.length; i++) {
        var chordSeqsAndFormat = []
        for (var j=0; j<subChords[i].length; j++) {
            subChordInd = subInds[i].indexOf(j)
            if (subChordInd == -1) { 
                chordSeqsAndFormat.push([subChords[i][j], false])
            } else {
                chordSeqsAndFormat.push([subChords[i][j], true])
            }
        }
        makeSuggestionItem(i, chordSeqsAndFormat)
    }
}

function updateSuggestionsSimple(chordSeqs, startInd) {
    for (var i=0; i<chordSeqs.length; i++) {
        var chordSeqsAndFormat = []
        for (var j=0; j<chordSeqs[i].length; j++) {
            chordSeqsAndFormat.push([chordSeqs[i][j], false])
        }
        makeSuggestionItem(startInd + i, chordSeqsAndFormat)
    }
}

function makeSuggestionItem(i, chordSeqsAndFormat) {
    parent = $('#suggest'+i.toString())
    parent.empty()
    var lineText = ''
    for (var j=0; j<chordSeqsAndFormat.length; j++) {
        var spanText = chordSeqsAndFormat[j][0] + ' '  
        lineText += spanText
        var span = $('<span>').text(spanText).css({"font-family":"monospace", "font-size":16}).appendTo(parent);
        if (chordSeqsAndFormat[j][1]) { span.css("font-weight", "bold") }
    }
    suggestTexts[i] = lineText
    var button = $('<button>').addClass('btn btn-info btn-xs').text('Use').appendTo(parent)
    button.click(function(e) {
        inputText.val(lineText)
        //socket.emit("generateAlternatives", inputText.val(), inputText.val().length-1)
        // textChange also calls generateAlternatives
        socket.emit("textChange", inputText.val(), inputText.val().length-1, false)
    });    
    makePlayButton(parent, lineText)
};

// temporarily disabled
//posText = $('<p>').addClass('p').text('Current position: ').appendTo(parent);
posText = $('<p>').addClass('p').text('Current position: ')
posInt = $('<span>').text('').appendTo(posText);
//chordPickerPrompt = $('<p>').addClass('p').text('Selected chord:  ').appendTo(parent);
//chordLabelSpan = $('<span>').text('').css("font-family", "monospace").appendTo(chordPickerPrompt);

// TODO: not yet working
function makeStopPlayButton(parent, id) {
    var btn = $('<button>').addClass('btn').attr('id', id).text('Stop!').appendTo(parent)
    btn.click(function(e) {
        console.log('stop clicked', allNotes)
        //MIDI.stopAllNotes() //not working yet
        for (var i=0; i<allNotes.length; i++) {
            console.log('noteoff', allNotes[i])
            MIDI.noteOff(0, allNotes[i][0], allNotes[i][1]+0.1);
        }
        allNotes = []
    });
}

// associated with fix text
function makePlayButton(parent, text) {
    var btn = $('<button>').addClass('btn btn-xs').text('Play!').appendTo(parent);
    btn.click(function(e) {
        console.log('play text button clicked', text)
        socket.emit("playSeq", text, 0)
    });
}

// diff strings
function diffStrs(one, other) {
    var diff = JsDiff.diffChars(one, other);
    var p = $('<p>').addClass('p').css("font-family", "monospace").css("font-size", 12).css("margin", 2).appendTo(past);
    diff.forEach(function(part){
      // green for additions, red for deletions
      // grey for common parts
      var color = part.added ? 'green' :
        part.removed ? 'red' : 'grey';
      var span = document.createElement('span');
      span.style.color = color;
      span.appendChild(document.createTextNode(part.value));
      //display.appendChild(span);
      p.append(span)
    });
};

caretPos = undefined 
keyPressed = undefined
// display caret pos
inputText.bind("mouseup", 
    function() {
        caretPos = $(this).caret().start
        posInt.text(caretPos)
});

// display the past
inputText.keyup(function(e) { 
    var len = inputHistory.length
    var value = this.value
    //caretPosP = caretPos
    caretPos = inputText.caret().start
    posInt.text(caretPos)
    console.log('keyup', e.which, value, this.value[caretPos-1])
    
    socket.emit("generateAlternatives", value, caretPos)
    
    // detect shift enter
    if (e.which==13 && event.shiftKey) {
        console.log('shift enter')
        socket.emit("playSeq", this.value, inputText.caret().start)
    } 
    else if (e.which==40 && suggestInd < suggestList.length - 1) {
        console.log('suggest', suggestInd)
        $('#suggest'+suggestInd).css('background-color', 'white')
        suggestInd += 1
        $('#suggest'+suggestInd).css('background-color', '#eee')
    }
    else if (e.which==38 && suggestInd > -1) {
        console.log('suggest', suggestInd)
        $('#suggest'+suggestInd).css('background-color', 'white')
        suggestInd -= 1
        $('#suggest'+suggestInd).css('background-color', '#eee')
        
    }
    // arrow left, arrow right(39), back, del, shift(16)
    else if (value[caretPos-1] == ' ' && e.which!=37 && e.which!=8 && e.which!=46 && e.which!=16) {
        console.log('not shift enter')
        socket.emit("textChange", value, caretPos)//, caretPosP)
    };
});


function appendText(text) {
    if (inputText.val().length == 0) {
        inputText.val( text ) 
    } else {
        inputText.val( inputText.val() + " " + text ) 
    }
    socket.emit("generateAlternatives", inputText.val(), inputText.val().length-1)
};

