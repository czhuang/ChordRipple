var moduleId = 'myModule';

angular.module(moduleId, ['ui.bootstrap']);
// testing rating


function makeRating(id, parent) {
    var controllerId = 'rating_' + id

    angular.module(moduleId).controller(controllerId, ['$scope', function ($scope) {
        $scope.rate = 2;
        $scope.max = 7;
        $scope.isReadonly = false;
        $scope.hoveringOver = function(value) {
            $scope.overStar = value;
            $scope.percent = value;
        };
        
        $scope.$watch('rate', function(value) {
            console.log('rating', value, controllerId)
            var ind = controllerId.split('_').pop()
            var text = $('#bookmarkTextSpan_'+ind).text()
            console.log('socket', socket)
            socket.emit('rating', value, ind, text) 
            
        })

    }]);

    var ng_controller = $('<span>').attr('id', 'rating_'+id).attr('ng-controller', controllerId).addClass("ng-scope").appendTo(parent)
    ng_controller.attr('ng-hide', '1').css("margin-left", 6).css("margin-top", 5)
    var rate_model = $("<rating>").attr('ng-model', 'rate').attr({'max':"max", 'readonly':"isReadonly"})
    rate_model.attr({'on-hover':'hoveringOver(value)', 'on-leave':'overStar=null'}).appendTo(ng_controller)
    $('<span>').addClass('label').attr({'ng-class':"{'label-warning': percent<3, 'label-info': percent>=3 && percent<5, 'label-success': percent>=5}", "ng-show":"overStar && !isReadonly"}).text("{{percent}} stars").appendTo(ng_controller);
};


function makeLikeCross(id, parent) {
    var controllerId = 'likeCross' + id
    
    var clearButton = $('<button>').addClass('btn, btn-xs btn-default').text('clear').attr('ng-disabled', 'isReadonly')
    clearButton.css('line-height', 1.0)

    angular.module(moduleId).controller(controllerId, function ($scope) {
        $scope.like = 0;
        $scope.cross = 0;
        $scope.max = 1;
        $scope.isReadonly = false;
        
        $scope.$watch('like', function(value) {
            console.log('like', value)
        })
        
        $scope.$watch('cross', function(value) {
            console.log('cross', value)
        })
        
    });

    var ng_controller = $('<div>').addClass('ng-scope').attr('ng-controller', controllerId).appendTo(parent)
    
    // like rater
    var like_model = $("<rating ng-model='like'>").addClass('ng-scope').attr({'max':"max", 'readonly':"isReadonly"})
    like_model.attr({'state-on':"'glyphicon-heart'", 'state-off':"'glyphicon-heart-empty'"})    
    like_model.attr({'on-hover':'hoveringOver(value)', 'on-leave':'overStar=null'}).appendTo(ng_controller)
    
     // cross rater
    var cross_model = $("<rating ng-model='cross'>").addClass('ng-scope').attr({'max':"max", 'readonly':"isReadonly"})
    cross_model.attr({'state-on':"'glyphicon-remove-sign'", 'state-off':"'glyphicon-remove-circle'"})    
    cross_model.attr({'on-hover':'hoveringOver(value)', 'on-leave':'overStar=null'}).appendTo(ng_controller)
    
    clearButton.attr('ng-click', "cross=0; like=0").appendTo(ng_controller)

};



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
  
  socket.on("updateChordSuggestions", function(subChords, subInds, suggestionTypes) {
    updateSuggestions(subChords, subInds, suggestionTypes, '') 
  });
  
  socket.on("updateChordSuggestionsAbove", function(subChords, subInds, suggestionTypes) {
    console.log('----updateChordSuggestionsAbove')
    updateSuggestions(subChords, subInds, suggestionTypes, 'above') 
  });
  
  
  socket.on("updateHistory", function(previousInput, currentInput) {
    // update history display
    // temporarily disable
    if (false){
    if (previousInput != currentInput) {
        inputHistory.push(currentInput)
        diffStrs(previousInput, currentInput)
    };
    }
  });

  socket.on("dataLabels", function(receivedData, receivedTextLabels) {
    vertices = receivedData
    textLabels = receivedTextLabels
    console.log('...received', vertices, textLabels)
    if (!withoutPicker){
        setupPicker(vertices, textLabels, socket)
    }
  });
  
  socket.on("setInput", function(text) {
    setInputText(text);
  });
  
  for (var i=0; i<maxNumSaveEntry; i++) {
    makeSaveEntry(i);
  }
  angular.element(document).ready(function() {
    angular.bootstrap(document, [moduleId]);
  });

}); //end for document ready


// setup UI
var widthPercent = '85%' 

// past / bookmark region
past = $('#past')
var bookmarkTitle = $('<h5>').appendTo(past)
var bookmarkPanel = $('<div>').attr('id', 'bookmarkPanel').css({'height':'100px', 'overflow':'auto', 'width':'720px'}).appendTo(past)

// main
var parent = $('#chart')

// added in html instead as an outter div
//var ng_app = $('<div>').addClass('ng-scope').attr('ng-app', moduleId).appendTo(parent)

// $('<br>').appendTo(parent)
var taskMode = false
if (!taskMode) {
    var headerText = 'ChordRipple '
    var subheaderText = "Change one chord, and see the changes propagate down the list.  See substitutions, what's next, surrounding chords change, and continuations that lead you to the end of a phrase. "
    var header = $('<h1>').text(headerText).css('width', widthPercent).appendTo(parent)
    var subheader = $('<small>').text(subheaderText).appendTo(header)
} else {
    var promptInd = 3

    // task prompt
    if (promptInd == 0) {
        headerText = "Ringtone for a beaver?  Try generating as many creative and different chord sequences as possible. They could range from 6 to 10 chords long. "
        subheaderText = "For extra-musical inspiration, you can think of each sequence as a ringtone for a different pet, ringtone for a beaver, for a rubber duck, etc. You don't have to tell us what it was for. "
    } else if (promptInd == 1) {
        headerText = "Ringtone for your toothpaste?  Try generating as many creative and different chord sequences as possible. They could range from 6 to 10 chords long. "
        subheaderText = "We'll help you out by giving you some chords to try out on the way. :)   For extra-musical inspiration, you can think of each sequence as a ringtone for a different household items, ringtone for your toothpaste, for your curtains, etc. You don't have to tell us what it was for. "
    } else if (promptInd == 2) {
        headerText = "Ringtone for somebody you know?  Try generating as many creative and different chord sequences as possible. They could range from 6 to 10 chords long. "
        subheaderText = "We'll help you out by giving you some chords to try out on the way. :)   For extra-musical inspiration, you can think of each sequence as a ringtone for a different friend, ringtone for your next-door neighbour, for your imaginary twin, etc. You don't have to tell us what it was for. "
    } else {
        headerText = "Let's break out of our habits!  Try generating as many creative and different chord sequences as possible. They could range from 6 to 10 chords long. "
        subheaderText = "We'll help you out by suggesting some chords on the way. :)"
    }

    var header = $('<h3>').text(headerText).css('width', widthPercent).appendTo(parent)
    var subheader = $('<small>').text(subheaderText).appendTo(header)
}

$('<br>').appendTo(parent)



// temporarily disable
// toggle UI for symbol type
//var symbolTogglePanel = $('<p>').text('Choose symbol type: ').appendTo(parent)
//var symbolToggle = $('<div>').addClass("btn-group").attr("data-toggle", "buttons").appendTo(symbolTogglePanel)
//makeRadioButton(symbolToggle, 'I', 'options', 'roman', true)
//makeRadioButton(symbolToggle, 'C', 'options', 'letter', false)


// add playback speed slider
var speedPanel = $('<p>').text('Playback speed (chords per minute): ').appendTo(parent)
var initialSpeed = 92
//$('<div>').attr('id', 'slider').addClass('.slider .slider-horizontal').appendTo(parent)
var sliderSpan = $('<span>').text(initialSpeed).appendTo(speedPanel)
var slider = $('<div>').slider({min:20, step:1, value:initialSpeed, max:220}).addClass('slider slider-horizontal').css({'width': '20%'}).appendTo(parent)

slider.slider({
    slide: function(event, ui){
        displayValue = ui.value.toFixed(0)
        sliderSpan.text(displayValue);
        sliderHandlerDebounced(ui.value)
}});
        
sliderHandlerDebounced = _.debounce(function(speed){
    socket.emit('setPlaybackSpeed', speed)}, 300);

        

// main input UI
//var inputTextInstructText = 'Type chords here as roman numerals (i.e. I ii IV).  Letter chords to come soon, and better voicing of chords.  Right now chords are played as a block. '
// var inputTextInstructText = "As soon as you come up with something interesting, you can click the blue 'Save' button.  You'll be able to choose your best ones later."
//var inputTextInstructText = "Type chords here (i.e. C Am F G7).  As soon as you come up with something interesting, you can click the blue 'Save' button.  You'll be able to choose your best ones later."
//var inputTextInstructText = "Type chords here (i.e. I V/V V IV6 vii/o7 I).  As soon as you come up with something interesting, you can click the blue 'Save' button.  You'll be able to choose your best ones later."
//var inputTextInstruct = $('<h5>').text(inputTextInstructText).appendTo(parent)



inputText = $('<input>').addClass('text').attr('id', 'inputtext').css({"font-family":"monospace", "font-size":16, "margin-top":8}).width(widthPercent).appendTo(parent);
inputText.css({"width":'600px'}) 
inputText.focus(function(e) {
    if (inputText.val().length == 0) {
        socket.emit("startSeqs")
    }
});

var inputSaveButton = $('<button>').addClass('btn btn-primary').text('Save').appendTo(parent)
inputSaveButton.click(function(e) {updateSaveEntry(e)});

var maxNumSaveEntry = 20;
var numSaved = 0;

var maxNumSuggestions = 20;
var maxAboveNumSuggestions = maxNumSuggestions;


var inputTextPlayButton = $('<button>').addClass('btn').text('Play!').appendTo(parent);
inputTextPlayButton.click(function(e) {
    socket.emit("playSeq", inputText.val(), inputText.caret().start, 'user', undefined, undefined)
});

var loopButton = $('<button>').addClass('btn').text('Loop! (Repeats once)').css('border-radius', '12px').appendTo(parent);
loopButton.click(function(e) {
    socket.emit("playSeq", inputText.val(), inputText.caret().start, 'user', undefined, undefined, true)
});

var clearButton = $('<button>').addClass('btn btn-primary').text('clear').appendTo(parent)
clearButton.click(function(e) {
    text = inputText.val()
    inputText.val('')
    clearSuggestions()
    socket.emit('clear', text) 
});
    

createSuggestions(parent, "above")
createSuggestions(parent, '')


function makeRadioButton(parent, text, name, id, active) {
    var label = $('<label>').addClass('btn btn-primary').text(text).appendTo(parent)
    if (active) { 
        label.addClass('active')
    }
    var input = $('<input>').attr('type', 'radio').attr('name', name).attr('id', id).attr('autocomplete', 'off').appendTo(label)

    label.click(function(e) {
        console.log('setSymbolType', inputText.val(), id)
        socket.emit('setSymbolType', inputText.val(), id)
    });
}

function updateSaveEntry() {
    if (numSaved == 0) {
        bookmarkTitle.text('Saved chord sequences:')
    }
    var lineText = inputText.val()
    socket.emit("inputSave", inputText.val())
    
    // make save entry widgets visible, and add the buttons that need lineText
    $('#bookmarkTextSpan_'+numSaved).text(lineText).attr('display', 'inline')
    var p = $('#bookmark'+numSaved)
    p.attr('display', 'inline')
    
    var ratingWidget = p.children().last()
    // put here because need text in closure
    //bookmark entry buttons
    var button = $('<button>').addClass('btn btn-info btn-xs').text('Use').insertBefore(ratingWidget)
    button.click(function(e) {
        inputText.val(lineText)
        socket.emit("textChange", inputText.val(), inputText.val().length-1, "use", "user", numSaved, false)
        //socket.emit("playSeq", lineText, 0, 'user', undefined, undefined, false)
    })
    var playButton = makePlayButton(p, lineText, 'input', numSaved, false)
    playButton.insertBefore(ratingWidget)

    $('#rating_'+numSaved).removeClass('ng-hide').attr('ng-show', '1')
    
    bookmarkPanel.scrollTop((p.height()+10) * numSaved) 
    numSaved += 1     
}

function makeSaveEntry(ind) {
    var p = $('<p>').attr('id', 'bookmark'+ind).appendTo(bookmarkPanel)
    p.attr('display', "none");
    var span = $('<span>').attr('id', 'bookmarkTextSpan_'+ind).text('').css({"font-family":"monospace", "font-size":16}).appendTo(p);
    makeRating(ind, p) 
}   


function setInputText(text) {
    inputText.val(text)
}


function createSuggestions(parent, id) {
    for (var i=0; i<maxNumSuggestions; i++) {
        console.log('creating suggestions', 'suggestion'+id+i)
        var p = $('<p>').attr('id', 'suggest'+id+i)
        if (id == 'above') {
            var inputTextTemp = $('#inputtext')
            console.log('createSuggestions', id, inputTextTemp)
            p.insertBefore(inputTextTemp)
        } else {
            p.appendTo(parent)
        }
    }
}

function clearSuggestions() {
    for (var i=0; i<maxNumSuggestions; i++) {
        parent = $('#suggest'+i.toString())
        parent.empty()
        parent = $('#suggest'+'above'+i.toString())
        parent.empty()
    }
}

function makeSuggestionSubroutine(subChords, subInds, i, idStr) {
        var chordSeqsAndFormat = []
        for (var j=0; j<subChords[i].length; j++) {
            subChordInd = subInds[i].indexOf(j)
            if (subChordInd == -1) { 
                chordSeqsAndFormat.push([subChords[i][j], false])
            } else {
                chordSeqsAndFormat.push([subChords[i][j], true])
            }
        }
        makeSuggestionItem(chordSeqsAndFormat, i, idStr)
}

function updateSuggestions(subChords, subInds, suggestionTypes, id) {
    console.log('--updateSuggestions', id)
    if (id != 'above') {
        clearSuggestions()
    }
    
    var maxNum = subChords.length
    if (subChords.length > maxNumSuggestions) {
        maxNum = maxNumSuggestions
    }

    for (var i=0; i<maxNum; i++) {
        if (id != 'above') {
            var idStr = i.toString()
        } else {
            var idStr = 'above'+ (maxNum-i).toString()

        }
        makeSuggestionSubroutine(subChords, subInds, i, idStr)
    }
}


function makeSuggestionItem(chordSeqsAndFormat, i, idStr) {
    parentStr = '#suggest'+idStr
    parent = $(parentStr)
    console.log('makeSuggestionItem', parentStr)
    parent.empty()
    var lineText = ''
    //var onlyLastChanged = true
    for (var j=0; j<chordSeqsAndFormat.length; j++) {
        var spanText = chordSeqsAndFormat[j][0] + ' '  
        lineText += spanText
        var span = $('<span>').text(spanText).css({"font-family":"monospace", "font-size":16}).appendTo(parent);
        if (chordSeqsAndFormat[j][1]) { 
            span.css("font-weight", "bold") 
            // if not last index but changed
            //if (j < chordSeqsAndFormat.length-1 && chordSeqsAndFormat[j][1]) {
            //    onlyLastChanged = false
            //} 
        }
    }
    makeUseButton(parent, lineText, chordSeqsAndFormat, i, idStr)
    makePlayButton(parent, lineText, 'machine', i, idStr, true)
    makePlayChangeButton(parent, chordSeqsAndFormat, 'machine', i, idStr)
};

// temporarily disabled
//posText = $('<p>').addClass('p').text('Current position: ').appendTo(parent);
posText = $('<p>').addClass('p').text('Current position: ')
posInt = $('<span>').text('').appendTo(posText);
//chordPickerPrompt = $('<p>').addClass('p').text('Selected chord:  ').appendTo(parent);
//chordLabelSpan = $('<span>').text('').css("font-family", "monospace").appendTo(chordPickerPrompt);

function makeUseButton(parent, lineText, chordSeqsAndFormat, i, idStr){
    var button = $('<button>').addClass('btn btn-info btn-xs').text('Use').appendTo(parent)
    button.click(function(e) {
        inputText.val(lineText)
        //socket.emit("generateAlternatives", inputText.val(), inputText.val().length-1)
        // textChange also calls generateAlternatives
        //console.log('textChange, onlyLastChanged', onlyLastChanged)
        socket.emit("textChange", lineText, inputText.val().length-1, "use", "machine", i, idStr, false)
        // false: don't log this playback
        socket.emit("playSubseq", chordSeqsAndFormat, 'dont_log', i, idStr, false)
    });   
}

// associated with fix text
function makePlayButton(parent, text, author, i, idStr, doAppend) {
    var btn = $('<button>').addClass('btn btn-xs').text('Play!')
    if (doAppend) {btn.appendTo(parent)};
    btn.click(function(e) {
        console.log('play text button clicked', text)
        socket.emit("playSeq", text, 0, author, i, idStr)
    });
    return btn
}

function makePlayChangeButton(parent, chordSeqsAndFormat, author, i, idStr) {
    var btn = $('<button>').addClass('btn btn-xs').text('Play bold!').css('border-radius', '8px').appendTo(parent);
    btn.click(function(e) {
        console.log('playSubseq button clicked', i)
        socket.emit("playSubseq", chordSeqsAndFormat, author, i, idStr)
    });
}



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
        socket.emit("textChange", value, caretPos, "edit", "user", undefined)//, caretPosP)
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

