

function status(x) {
  console.log('status', x);
  $('#status').text(x);
}


var withoutPicker = true
var demo = true


var inputHistory = [];
var vertices = [];
var labels = [];
var ratingWidgets = [];

// d3 picker interface...
var w = 750, 
    h = 750;

//var WIDTH = 680
var instructionTextSize = 'h6'
var instructionStepSize = 'h4'
var MARGIN = "3px"
var looping = false  // should be Lopping Mode
var playingSubseq = false
var simultaneouslyPlaybackRequested = false
var queryState = undefined
//var suggestionPanelActiveId = undefined
//var itemIdx = undefined
var ordering = undefined
var experiment_names = ['single_subs', 'single_subs', 'ripple']


$(document).ready(function() {

  // playback
//  socket.on("playNotes", function(notes) {
//    console.log('playNotes', notes)
//    song = makeSong(notes, sequencer)
//    var loop = false
//    startChordsPlaybackCallback(loop)
////    playMidiDebounced(midiNotes)
//  });

  socket.on("playSeq", function(notes, play, query) {
//    queryState = undefined
    playingSubseq = false
    userFocusedCell = false

    console.log('--- playSeq updating sequence', notes)
    song_main = makeSong(notes, sequencer)
    // this happens before song_main is assigned to song
    resetSong()
    queryState = query

    song = song_main
    playbackStopTime = ticksToSecond(song.durationTicks)

    if (simultaneouslyPlaybackRequested) {
        startPlaybackCallback()
    } else if (play != false && !looping) {
        startSongPlaybackCallback()
    }
    //playMidiSeqDebounced(noteSeqs, durs)
  });

  socket.on("playSubseq", function(notes, query) {
    playingSubseq = true
    console.log('playSubseq', notes)

    if (looping) {
        // don't do anything
        console.log('playSubseq: not doing anything')
    } else if (notes.length == 0) {
        song = undefined
        playbackStopTime = 0
        console.log('playSubseq song', song)
    } else {
        // need to be have the notes for all chords
        resetSong()
        song = makeSong(notes, sequencer)
        console.log('song length', ticksToSecond(song.durationTicks))
        if (song_main != undefined) {
            console.log('song main length', ticksToSecond(song_main.durationTicks))
        }
        playbackStopTime = ticksToSecond(song.durationTicks)
//
//        // get song offset
//        chordSeqsAndFormat = query.chordSeqsAndFormat
//        var offsetChordIdx = 0
//        for (var i=0; i<chordSeqsAndFormat.length; i++) {
//            if (chordSeqsAndFormat[i][1]) {
//                offsetChordIdx = i
//                break
//            }
//        }
//
//        var endChordIdx = offsetChordIdx
//        for (var i=offsetChordIdx; i<chordSeqsAndFormat.length; i++) {
//            if (!chordSeqsAndFormat[i][1]) {
//                endChordIdx = i
//                break
//            }
//        }
//        var offsetTime = videoScore.getTimeFromEventIdx(offsetChordIdx)
//        playbackStopTime = videoScore.getTimeFromEventIdx(endChordIdx)
//        console.log('...playSubseq offsetChordIdx, offsetTime', offsetChordIdx, offsetTime)
//        console.log('...playSubseq endChordIdx, playbackStopTime', endChordIdx, playbackStopTime)

        queryState = query
        var loop = false
        updateMidiSliderFlag = false
        startSongPlaybackCallback(loop)//, offsetTime)
        //playMidiSeqDebounced(noteSeqs, durs)
    }

  });

  socket.on("ordering", function(order) {
    ordering = order
    console.log('ordering', ordering)
  })

  socket.on("set_seq", function(lineText) {
    console.log('set_seq', lineText)
    inputText.val(lineText)
  })

  socket.on("updateChordSuggestions", function(subChords, subInds, suggestionTypes) {
    console.log('----updateChordSuggestions', subChords.length)
    // TODO: added below to suggestion panel id
    updateSuggestions(subChords, subInds, suggestionTypes, '')
    $(window).scrollTop(tempScrollTop);
  });

  socket.on("updateChordSuggestionsAbove", function(subChords, subInds, suggestionTypes) {
    console.log('----updateChordSuggestionsAbove', subChords.length)
    updateSuggestions(subChords, subInds, suggestionTypes, 'above')
    $(window).scrollTop(tempScrollTop);
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

  //TODO: angular related
//  angular.element(document).ready(function() {
//    angular.bootstrap(document, [moduleId]);
//  });


   socket.on("survey", function(questions) {
    questionsTextArea.text(questions[0])
   })


   $('#videoInspiration').hide()
   $('#playbackControls').hide()

}); //end for document ready


// setup UI
var widthPercent = '85%'
var WIDTH_PIXEL = '720px'
var WIDTH_PIXEL = '700px'

// past / bookmark region
past = $('#past')
//$('<hr>').appendTo(past)
var bookmarkTitle = $('<p>').appendTo(past)
var bookmarkPanel = $('<div>').attr('id', 'bookmarkPanel').appendTo(past) //.css({'height':'200px', 'overflow':'auto', 'width':WIDTH_PIXEL})

function clearBookmarks() {
//    bookmarkTitle.text('')
    for (var i=0; i<numSaved; i++) {
//        $('#bookmark'+i).empty()
        $('#bookmarkTextSpan_'+i).text('')
    }
    numSaved = 0
//    bookmarkPanel.empty()
}

if (!demo) {
$('<hr>').appendTo(past)
}
var highlightColor = getColorForFocus()

var stepFour = $('<'+instructionStepSize+'>').text('Step 3: ').css('color', highlightColor)

if (!demo) {
    stepFour.appendTo(past)
}

if (!demo) {
//var feedbackPrompt = $('<small>').text("Do you have any comments about the experience? Bug reports?  Any features you wish was there?")
//var questionText = "How would you describe the chord transformations that you performed?  What was your goal?  Did you have a clear goal in mind at the beginning?  If not, when and how did the goal emerge?  How would you describe your exploration process? When did you use ripples (if they were available)?"
var questionText = "How did you use the chord recommendations to transform the chord progression?  What was your goal?  Was the goal clear from the beginning?  If not, when and how did the goal emerge?  How would you describe your exploration process?"
var questionTextTwo = "\nWhen and how did you use ripples (if they were available)?  If you are in the second or third condition, how are the chord recommendations in this condition different from the previous conditions? If you are in the third (last) condition, rank all the conditions with your most preferred first (i.e. 3 > 1 > 2).  Thank you!"
var feedbackPrompt = $('<small>').text(questionText)
feedbackPrompt.appendTo(stepFour)
$('<br>').appendTo(stepFour)
$('<p>').appendTo(stepFour)
var feedbackPromptTwo = $('<small>').text(questionTextTwo)
feedbackPromptTwo.appendTo(stepFour)
var feedbackInputBox = $('<textarea>').css('width', WIDTH_PIXEL).css('height', '100px').appendTo(past)

var userFeedbackPanel = $('<'+instructionStepSize+'>').text('Step 4: ').css('color', highlightColor)
}
if (!demo) {
    userFeedbackPanel.appendTo(past)
}


// TODO: add subtitles to questions?
questions = [['I am experienced in composing.  ', 'Strongly disagree', 'Strongly agree'],
['Chords are important in my compositions. ', 'Strongly disagree', 'Strongly agree'],
['I am fluent on an instrument that is natural for playing chords.  ', 'Strongly disagree', 'Strongly agree', true],

    ['I wanted to come up with something different from what I would normally do.  ', 'Strongly disagree', 'Strongly agree'],
    ['How early in the process did I know what I wanted to express.  ', 'In the beginning', 'At the end'],
    ['I focused on coming up with:  ', 'one chord progression', 'multiple different chord progressions.', true],

    ['I wanted to explore a broad range of chords.  ', 'Strongly disagree', 'Strongly agree'],
    ['The tool helped me to use a wider range of chords then I would normally do.  ', 'Strongly disagree', 'Strongly agree', true],

    ['The chord recommendations:  ', 'hindered', 'inspired me in working out what I wanted to express.'],
    ['The chord recommendations were:  ', 'irrelevant', 'relevant'],
    ['The chord recommendations were:  ', 'too limited', 'too broad', true],

    ['Ripples made it easier to explore different ideas.  ', 'Strongly disagree', 'Strongly agree'],
    ['Ripples made it easier to try out difficult chords.  ', 'Strongly disagree', 'Strongly agree', true],

    ['The tool made me compose:  ', 'slower', 'faster'],
    ['With the tool, I composed chord progressions that were: ', 'less creative', 'more creative than what I would come up with in a short amount of time.'],
    ['I was happy with the chord progressions I came up with.  ', 'Strongly disagree', 'Strongly agree', true],

    ['How would you rate the overall experience with this current tool:  ', 'Poor', 'Excellent'],
    ["Would you consider using this tool for songwriting?  ", 'No at all', 'Absolutely yes']
]

for (var i=0; i<questions.length;i++) {
    var overAllPanel = $('<div>').appendTo(userFeedbackPanel)
    var overAllText = questions[i][0]
    $('<small>').text(overAllText).appendTo(overAllPanel)
    $('<small>').text(questions[i][1]).appendTo(overAllPanel)
    rating = makeRating(overAllPanel, overAllText, i)
    $('<small>').text(questions[i][2]).appendTo(overAllPanel)
    ratingWidgets.push(rating)
    if (questions[i].length > 3 && questions[i][3]) {
        $('<br>').appendTo(userFeedbackPanel)
    }
}




// email
//var emailPanel = $('<div>').appendTo(userFeedbackPanel)
//$('<small>').text("Do you want us to keep you posted with our later versions?  If so, please include your email in the comment box too:").appendTo(emailPanel)




if (!demo) {
    var feedbackButton = $('<button>').addClass('btn btn-primary').text('Next').appendTo(past)
    var questionsTextArea = $('<normal>').appendTo(past)


var experiment_names = ['tutorial', 'first', 'second', 'third']
var experiment_count = 0

feedbackButton.click(function(e) {
    var text = feedbackInputBox.val()
    socket.emit('comments', text)
    experiment_count += 1
    imageNames = ["bunny.jpg", "kandinsky.jpg", "pollock.jpg", "van_gogh.jpg"]

    if (experiment_count == 3) {
        feedbackButton.text("Submit")
    } else if (experiment_count > 3) {
        var thankYouPanel = $('<p>').css('margin-top', "10px").appendTo(past)
        var thankYouText = $('<normal>').text("Thank you for your feedback!  Hope you had fun playing with ChordRipple!")
        thankYouText.css('color', highlightColor).appendTo(thankYouPanel)
    }

    if (experiment_count >= imageNames.length ) {
       experiment_count = imageNames.length - 1
    }
    imageName = imageNames[experiment_count]
    $('#imageInspiration').attr('src', "js/images/" + imageName)


    // clears suggestions
    socket.emit('next', experiment_count)
    // clear text, and saved, and put in new questions
    clearInputText()
    clearBookmarks()
    feedbackInputBox.val('')
    clearRating()
    if (experiment_count == 3) {
        // add in ranking

        // make list of names
        var condition_names = []
        for (var i=0; i<ordering.length; i++) {
            exp_ind = ordering[i]
            var name = str(i) + ' (' + experiment_names[exp_ind] + ')'
            condition_names.push(exp_ind)
        }
        var rankingDiv = $('<div>').appendTo(userFeedbackPanel)

        $('<span>').text('>').appendTo(rankingDiv)

        $('<span>').text('>').appendTo(rankingDiv)


    }


}); // feedbackButton
    
} // if (!demo)

// -------- main -----------
var parent = $('#title')


// added in html instead as an outter div
//var ng_app = $('<div>').addClass('ng-scope').attr('ng-app', moduleId).appendTo(parent)

// $('<br>').appendTo(parent)
var taskMode = false
if (!taskMode) {
    var wantHeader = false
    if (wantHeader) {
        var headerText = 'ChordRipple v2.0 '
        //var subheaderText = "Change one chord, and see the changes propagate down the list.  See substitutions, what's next, surrounding chords change, and continuations that lead you to the end of a phrase. "
    //    var subheaderText = "Now with video!  Choose a chord to go with a scene in the video, and see how your choices affect the surrounding chords.  See a list of auto-completions that gives you substitutions for the current chord, what's around it, and suggestions for what's next.  "
    //    var subheaderText = "Let's sketch a few chords to go with the video! "
    //    var subheaderText = "We're currently updating so things might not be working. Sorry for the inconvenience... Please come back tomorrow after 11am.  Thanks!"
        var subheaderText = "Welcome!  You're about to try out a chord recommendation tool!  We hope you'll have some fun too!"

        var header = $('<h1>').text(headerText).css('width', WIDTH+'px').appendTo(parent)
        header.css('margin-bottom', '2px')
        var subheader = $('<small>').text(subheaderText).appendTo(header)
        var remainderText = "\nTo start, follow steps 1, 2, 3. Make sure your window is at least 1150px wide."
    //    var remainder = $('<small>').text(remainderText).appendTo(header)

        var highlightColor = getColorForFocus()
        var remainder = $('<h4>').text(remainderText).css('color', highlightColor)
        remainder.css('font-weight', 'lighter').css('margin-top', '0px').appendTo(parent)
    //    var reremainderText = "Make sure your window is at least 1150px wide."
    //    var reremainder = $('<h3>').text(reremainderText).appendTo(remainder)
    }
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



var parent = $('#chart')


function makeRating(parent, lineText, id, sz) {
    // rating can be for saved sequences or
    // for rating questions
    if (sz == undefined) {
        sz = 'xs'
    }
    var rating = $('<input>').attr('type', 'number').addClass('rating')
//    ratingWidgets.push(rating)
    rating.attr('min', 0).attr('max', 5).attr('step', 0.5)
    rating.attr('data-size', sz)
    rating.css({"display":"inline-block"})
    rating.appendTo(parent)
    rating.rating({'showClear': false, 'showCaption': false})

    rating.on('rating.change', function(event, value, caption) {
        console.log('lineText', lineText)
        console.log(value);
        console.log(caption);
        socket.emit('rating', lineText, id, value, caption)
    });
    return rating
}

function clearRating() {
    $('#bookmarkPanel').find('.btn').remove()
    $('#bookmarkPanel').find('.star-rating').remove()


    console.log('ratingWidgets', ratingWidgets.length)
    for (var i=0; i<ratingWidgets.length; i++) {
        ratingWidgets[i].val(0)
    }
//    $('.rating').val(0)
}

// temporarily disable
// toggle UI for symbol type
//var symbolTogglePanel = $('<p>').text('Choose symbol type: ').appendTo(parent)
//var symbolToggle = $('<div>').addClass("btn-group").attr("data-toggle", "buttons").appendTo(symbolTogglePanel)
//makeRadioButton(symbolToggle, 'I', 'options', 'roman', true)
//makeRadioButton(symbolToggle, 'C', 'options', 'letter', false)


// add playback speed slider
var SHOW_TEMPO_SLIDER = false
if (SHOW_TEMPO_SLIDER) {
    var speedPanel = $('<p>').text('Playback speed (chords per minute): ').appendTo(parent)
    speedPanel.css('margin-top', "10px")
    speedPanel.css('margin-bottom', "2px")
    var initialSpeed = 92
    var initialSpeed = 60
    //$('<div>').attr('id', 'slider').addClass('.slider .slider-horizontal').appendTo(parent)
    var sliderSpan = $('<span>').text(initialSpeed).appendTo(speedPanel)
    var slider = $('<div>').slider({min:20, step:1, value:initialSpeed, max:220}).addClass('slider slider-horizontal').css({'width': '20%'}).appendTo(speedPanel)
    $('<hr>').appendTo(speedPanel)
    slider.slider({
        slide: function(event, ui){
            displayValue = ui.value.toFixed(0)
            sliderSpan.text(displayValue);
            sliderHandlerDebounced(ui.value)
    }});

    sliderHandlerDebounced = _.debounce(function(speed){
        socket.emit('setPlaybackSpeed', speed)}, 300);
}



// main input UI
//var inputTextInstructText = 'Type chords here as roman numerals (i.e. I ii IV).  Letter chords to come soon, and better voicing of chords.  Right now chords are played as a block. '
// var inputTextInstructText = "As soon as you come up with something interesting, you can click the blue 'Save' button.  You'll be able to choose your best ones later."
//var inputTextInstructText = "Type chords here (i.e. C Am F G7).  As soon as you come up with something interesting, you can click the blue 'Save' button.  You'll be able to choose your best ones later."
//var inputTextInstructText = "Type chords here (i.e. I V/V V IV6 vii/o7 I).  As soon as you come up with something interesting, you can click the blue 'Save' button.  You'll be able to choose your best ones later."
//var inputTextInstruct = $('<h5>').text(inputTextInstructText).appendTo(parent)

//var timings = [0.0, 0.5, 1.5, 3.3, 3.9, 5.3, 5.78]
//var videoScore = new VideoScore(timings)
console.log('videoScore', videoScore.syncEvents)
//videoScore.addEvent(0.0, 0)
//videoScore.addEvent(4.5, 1)
console.log('videoScore', videoScore)
var parentId =  "inputTextParent"
//parent.css({position: 'relative'})


var highlightColor = getColorForFocus()
var stepTwo = $('<'+instructionStepSize+'>').text('Step 2: ').css('color', highlightColor)

if (!demo) {
    stepTwo.appendTo(parent)
}

//var chordInstructText = "Type / choose some chords here.  Each input box takes a chord and corresponds to a scene in the video.  There's some auto-completions to help you along the way."
var chordInstructText = "Transform the given chord progression so that it works better with the image.  Feel free to explore the suggestions given and save as soon as you find something good and as often as you want (as there is currently no undo button).  All recommendations assume that the chord progression is in C.  You can modulate to other key areas but bear in mind that they will be relative to C. "
$('<small>').text(chordInstructText).appendTo(stepTwo)

// might have to start a panel for this
//var iFeelLuckyPanel = $('<p>').css('margin-top', '10px').appendTo(stepTwo)
//var additionText = "We can also start you off with a complete sequence: "
//var additionalLine = $('<small>').text(additionText).appendTo(iFeelLuckyPanel)
//var iFeelLuckyButton = $('<button>').addClass('btn btn-info btn-mini').text('I feel lucky')
//iFeelLuckyButton.appendTo(iFeelLuckyPanel)
//iFeelLuckyButton.click(function(e) {
//    socket.emit("generate_complete_seq")
//})



var inputTextParent = $("<p>").attr('id', parentId).css('margin-top', '12px').appendTo(parent)
//inputTextParent.css({'position': 'absolute'})
var parentPositionTop = parent.position().top
var desiredPositionTop = 437
console.log("parentPositionTop", parentPositionTop)
//inputTextParent.css({'top': desiredPositionTop-parentPositionTop})
inputTextParent.css({'offset': desiredPositionTop})

//insertBefore

var WIDTH = 680
var inputText = new InputTextCellSequence(parentId, 'input', videoScore, socket, WIDTH, 'user')
// didn't work somehow
//inputText.val('Cm F G Cm F G G7')

var suggestionsInputText = []
var suggestionsAboveInputText = []

$(document).ready(function() {
    console.log('--- socket ready: setting focus, keyup')
    inputText.setSocket(socket)
    // b/c calls focus, keyup inside
    console.log('videoScore', videoScore.syncEvents)
    inputText.setupCellsToVideoScore()
//    inputText.addEmptyCells(2, true)
//    inputText.focus() // setups the callback inside
//    inputText.keyup()

});

//inputText.focus(function(e) {
//    console.log('...focus', inputText.val().length)
//    if (inputText.val().length == 0) {
//        socket.emit("startSeqs")
//    }
//});

var inputSaveButton = $('<button>').addClass('btn btn-primary').text('Save').appendTo(inputTextParent)
inputSaveButton.click(function(e) {updateSaveEntry(e)});

var maxNumSaveEntry = 20;
var numSaved = 0;

var maxNumSuggestions = 20;
var maxAboveNumSuggestions = maxNumSuggestions;


var inputTextPlayButton = $('<button>').addClass('btn').text('Play').appendTo(inputTextParent);
inputTextPlayButton.click(function(e) {
    playSeq(inputText, 'user', 'play', true, false)

});



var inputTextStopButton = $('<button>').addClass('btn').text('Stop').appendTo(inputTextParent);
inputTextStopButton.click(function(e) {
    stopPauseSongPlaybackCallback(true)
});


function playSeq(lineTextObj, author, actionKind, play, loop, idStr, id) {
    var actionAuthor = 'user'
    var queryObject = new QueryObject(lineTextObj, author, actionKind, actionAuthor,
                                      play, loop, idStr, id, inputText.val())
//    queryState = queryObject
    console.log('inputTextPlayButton', queryObject)
    socket.emit("playSeq", queryObject)
}

//var loopButton = $('<button>').addClass('btn').text('Loop! (Repeats once)').css('border-radius', '12px').appendTo(inputTextParent);
//loopButton.click(function(e) {
//    socket.emit("playSeq", inputText.val(), inputText.caret().start, 'user', undefined, undefined, true)
//});

var clearButton = $('<button>').addClass('btn btn-primary').text('clear').appendTo(inputTextParent)
clearButton.click(function(e) {
    clearInputText()
});

function clearInputText() {
    console.log('...clear')
    text = inputText.val()
    inputText.val('')
    clearSuggestions()
    socket.emit('clear', text)
    // TODO: hack to make sure playback stops
    stopPauseSongPlaybackCallback(true)
    inputText.resetCellColorExceptIdx()

}


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

    // logging save action
    var lineText = inputText.val()
    socket.emit("inputSave", inputText.val())

    // updating save entry
    var p = $('#bookmark'+numSaved)
    p.attr('display', 'inline')

    // make save entry widgets visible, and add the buttons that need lineText
    var textSpan = $('#bookmarkTextSpan_'+numSaved)
    textSpan.text(lineText).attr('display', 'inline').appendTo(p)
    textSpan.css({'padding-right': '10px'})

    var button = $('<button>').addClass('btn btn-info btn-xs').text('Use').appendTo(p)
    button.click(function(e) {
        var originalText = inputText.val()
        inputText.val(lineText)
        makeUseButtonPartialHandler(lineText, 'user', 'use', 'user',
                                    undefined, undefined, originalText, undefined)
//        makeUseButtonPartialHandler('user', 'useSaved', 'user')

//        var queryObject = new QueryObject(inputText, 'user', 'useSaved', 'user',
//                                          false, false)
//        socket.emit("textChange", inputText.val(), inputText.val().length-1,
//                    "use", "user", numSaved, false)
        //socket.emit("playSeq", lineText, 0, 'user', undefined, undefined, false)
    })


    var playButton = makePlayButton(p, lineText, 'input', numSaved, false)
    playButton.css('margin-right', '8px').appendTo(p)

//    $('#rating_'+numSaved).removeClass('ng-hide').attr('ng-show', '1')
    var rating = makeRating(p, lineText, numSaved, 'sm')
    ratingWidgets.push(rating)

    bookmarkPanel.scrollTop((p.height()+10) * numSaved)
    numSaved += 1
}


function makeSaveEntry(ind) {
    var p = $('<p>').attr('id', 'bookmark'+ind).appendTo(bookmarkPanel)
    p.attr('display', "none");
    var span = $('<span>').attr('id', 'bookmarkTextSpan_'+ind).text('').css({"font-family":"monospace", "font-size":16}).appendTo(p);
    //TODO: angular related
    //makeRating(ind, p)

}


function setInputText(text) {
    inputText.val(text)
}


function createSuggestions(parent, id) {
    for (var i=0; i<maxNumSuggestions; i++) {
//        console.log('creating suggestions', 'suggestion'+id+i)
        var p = $('<p>').attr('id', 'suggest'+id+i)
        p.css('margin-bottom', "3px")
        if (i==0) {
            p.css('margin-top', '12px')
        }
        if (id == 'above') {
            var inputTextTemp = $('#inputTextParent')
//            console.log('createSuggestions', id, inputTextTemp)
            p.insertBefore(inputTextTemp)
        } else {
            p.appendTo(parent)
        }
    }
}


function clearSuggestions(id) {
//    for (var i=0; i<maxNumSuggestions; i++) {
//        var parent = $('#suggest'+id+i.toString())
//        // TODO: not yet worked out
//        var children = parent.children()
//        if (children.length != 0){
//            var originalLength = children.length
//            // assumes the only non inputs are at the end
//            for (var i=originalLength-1; i>=0; i--) {
//                if (children[i].type == "input") {
//                    children[i].value = ''
//                } else {
//                    console.log('...removing cells')
//                    children[i].remove()
//                }
//            }
//        }
//        var childrenOfTypeText = parent.children().filter('.text')
//        for (var i=0; i<childrenOfTypeText.length; i++) {
//            childrenOfTypeText[i].value = ''
//        }
//    }
    for (var i=0; i<maxNumSuggestions; i++) {
        parent = $('#suggest'+i.toString())
        parent.empty()
        parent = $('#suggest'+'above'+i.toString())
        parent.empty()
    }

    suggestionsInputText = []
    suggestionsAboveInputText = []
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
        var suggestionItem = makeSuggestionItem(chordSeqsAndFormat, i, idStr)
        return suggestionItem
}

function updateSuggestions(subChords, subInds, suggestionTypes, id) {
    // console.log('--updateSuggestions', id, subChords.length)
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
        var suggestInputText = makeSuggestionSubroutine(subChords, subInds, i, idStr)
        if (id != 'above') {
            suggestionsInputText.push(suggestInputText)
        } else {
            suggestionsAboveInputText.push(suggestInputText)
        }
    }
}


function makeSuggestionItem(chordSeqsAndFormat, i, idStr) {
//    parentStr = '#suggest'+idStr
    parentStr = 'suggest'+idStr
    parent = $('#'+parentStr)

    // count if is ripple, if ripple, then set margin to be -2px, otherwise 3px
    var changeCount = 0
    for (var i=0; i<chordSeqsAndFormat.length; i++) {
        if (chordSeqsAndFormat[i][1]) {
            changeCount += 1
        }
    }
    if (changeCount > 1) {
        parent.css('margin-bottom', '-2px')
    } else {
        parent.css('margin-bottom', '3px')
    }

//    console.log('makeSuggestionItem', parentStr, i, chordSeqsAndFormat)
    var suggestionInputText = new InputTextCellSequence(parentStr, 'div', videoScore, socket, WIDTH,
                                                        'machine', idStr, i)
    // by the time makeSuggestionItem, socket should already be available
    suggestionInputText.setSocket(socket)
    suggestionInputText.setupCellsToVideoScore()
    var widths = inputText.getWidths()
    var cells = suggestionInputText.getCells()
    var lineText = ''
    for (var j=0; j<widths.length; j++) {
        var spanText = ''
        if (j< chordSeqsAndFormat.length) {
            var spanText = chordSeqsAndFormat[j][0] + ' '
        }
        lineText += spanText
        var span = cells[j]
//        span.text(spanText).css({"font-family":"monospace", "font-size":16})//.appendTo(parent);
        span.css({"font-family":"monospace", "font-size":16})//.appendTo(parent);
        span.css({"display":"inline-block"})
        //span.attr('id', 'suggest'+idStr+'_'+j)
        var width = widths[j]
        if (j >= widths.length) {
            width = widths[widths.length-1]
        }

        if (j<chordSeqsAndFormat.length && chordSeqsAndFormat[j][1]) {
            span.css("font-weight", "bold")
//            span.val(spanText)
            span.text(spanText)
            // if not last index but changed
            //if (j < chordSeqsAndFormat.length-1 && chordSeqsAndFormat[j][1]) {
            //    onlyLastChanged = false
            //}
        } else {
//            slan.val('')
            span.text('')
        }

        var padding_extra = 3
        span.css({'width': width +'px'})
        // Not able to set width with span, but div should be good .css({'width': width +'px'})
        // div also don't need this, css({'padding': padding_extra+"px"}).
        // needed width could be negative, and this resets the padding set before for the right
//        var neededWidth = width - span.width() - 12
//        if (span.text() == '') {
//            neededWidth = width //+ 3*2
//        }
//
//        console.log('neededWidth', neededWidth, width, span.width())
//        span.css({'padding-right': neededWidth})

        span.css('border', 'None')
//        span.attr('readonly', true)
//
//        span.val(spanText)

    }
    makeUseButton(parent, lineText, chordSeqsAndFormat, i, idStr)
//    makePlayButton(parent, lineText, 'machine', i, idStr, true)
    makePlayChangeButton(parent, lineText, chordSeqsAndFormat, 'machine', i, idStr)
    // the last true value is for playContext which means also play context
    makePlayChangeButton(parent, lineText, chordSeqsAndFormat, 'machine', i, idStr, true)
    return suggestionInputText
}


function makeSuggestionItemBack(chordSeqsAndFormat, i, idStr) {
    parentStr = '#suggest'+idStr
    parent = $(parentStr)
    // console.log('makeSuggestionItem', parentStr)
    parent.empty()
    var lineText = ''
    //var onlyLastChanged = true
    var widths = inputText.getWidths()
//    for (var j=0; j<chordSeqsAndFormat.length; j++) {
    for (var j=0; j<widths.length; j++) {
        var spanText = ''
        if (j< chordSeqsAndFormat.length) {
            var spanText = chordSeqsAndFormat[j][0] + ' '
        }

        lineText += spanText
        var span = $('<input>').text(spanText).css({"font-family":"monospace", "font-size":16}).appendTo(parent);
        span.attr('id', 'suggest'+idStr+'_'+j)
        var width = widths[j]
        if (j >= widths.length) {
            width = widths[widths.length-1]
        }
        span.css({'padding': 3+"px"}).css({'width': width +'px'})
        //span.css('border', 'None')
        span.attr('readonly', true)
        span.val(spanText)

        if (j<chordSeqsAndFormat.length && chordSeqsAndFormat[j][1]) {
            span.css("font-weight", "bold")
            // if not last index but changed
            //if (j < chordSeqsAndFormat.length-1 && chordSeqsAndFormat[j][1]) {
            //    onlyLastChanged = false
            //}
        }
    }
    makeUseButton(parent, lineText, chordSeqsAndFormat, i, idStr)
    makePlayButton(parent, lineText, 'machine', i, idStr, true)
    makePlayChangeButton(parent, lineText, chordSeqsAndFormat, 'machine', i, idStr)
};

// temporarily disabled
//posText = $('<p>').addClass('p').text('Current position: ').appendTo(parent);
posText = $('<p>').addClass('p').text('Current position: ')
posInt = $('<span>').text('').appendTo(posText);
//chordPickerPrompt = $('<p>').addClass('p').text('Selected chord:  ').appendTo(parent);
//chordLabelSpan = $('<span>').text('').css("font-family", "monospace").appendTo(chordPickerPrompt);

function makeCompleteSeqFromChordSeqsAndFormat(chordSeqsAndFormat) {
    var lineText = ''
    var lineTextList = []
    for (var i=0; videoScore.getTimings().length; i++) {
        if (i >= chordSeqsAndFormat.length) {
            break
        }
        lineText += chordSeqsAndFormat[i][0] + ' '
        lineTextList.push(chordSeqsAndFormat[i][0])
    }
    console.log('lineText', lineText)
    console.log('lineTextList', lineTextList)
    return lineTextList
}

//function mergeOriginalWithUse(originalTextList, chordSeqsAndFormat) {
//    var lineText = ''
//    for (var i=0; videoScore.getTimings().length; i++) {
//        if (i >= chordSeqsAndFormat.length) {
//            break
//        }
//        if (chordSeqsAndFormat[i][1]) {
//            lineText += chordSeqsAndFormat[i][0] + ' '
//        } else {
//            lineText += originalTextList[i] + ' '
//        }
//    }
//    return lineText.trim()
//}

function makeUseButton(parent, lineText, chordSeqsAndFormat, i, idStr){
    var button = $('<button>').addClass('btn btn-info btn-xs').text('Use').appendTo(parent)
    button.click(function(e) {
        tempScrollTop = $(window).scrollTop();

        var originalText = inputText.getTextList()
        console.log('......originalText', originalText)
    //        // cuts out the extra chords
    //        lineText = inputText.val()

        // cuts out the extra chords here...
    //        var useText = mergeOriginalWithUse(originalText, chordSeqsAndFormat)
    //        console.log('...useText', useText)
    //        inputText.val(useText)


        if (chordSeqsAndFormat != undefined) {
            lineTextList = makeCompleteSeqFromChordSeqsAndFormat(chordSeqsAndFormat)
        }
    //        inputText.val(lineText)
        inputText.setCells(lineTextList)

        console.log('Use to inputText', lineTextList)

        makeUseButtonPartialHandler(lineTextList, 'machine', 'use', 'user',
                                    idStr, i, originalText, chordSeqsAndFormat)

})};


function makeUseButtonPartialHandler(lineTextList, author, actionKind, actionAuthor,
                                     idStr, i, originalText, chordSeqsAndFormat) {

    //socket.emit("generateAlternatives", inputText.val(), inputText.val().length-1)
    // textChange also calls generateAlternatives
    //console.log('textChange, onlyLastChanged', onlyLastChanged)
    lineTextObject = new LineTextObject(lineTextList,
                                        inputText.getDurations(),
                                        inputText.getFocusCellIdx())
    var play = true
    if (looping) {
        play = false
    }

    var queryObject = new QueryObject(lineTextObject, author, actionKind, actionAuthor,
                                      play, false, idStr, i,
                                      originalText)

    console.log('makeUseButton', queryObject)

    if (chordSeqsAndFormat != undefined) {
        queryObject.chordSeqsAndFormat = chordSeqsAndFormat
    }
    socket.emit("textChange", queryObject)


    inputText.updateFocus()

    //socket.emit("textChange", lineText, inputText.val().length-1, "use", "machine", i, idStr, false)

    // false: don't log this playback
    // don't playback use when if in looping mode
//        if (!looping) {
//            emitPlaySubseq(lineText, chordSeqsAndFormat, 'dont_log', i, idStr, false)
//        } else {
//            // don't playback but update song
//            // TODO: makes the lineTextObject and queryObject twice
//            console.log('Use while looping')
//            //updateSong(lineText, false, 'machine', idStr, i)
//        }

}

function emitPlaySubseq(text, chordSeqsAndFormat, author, i, idStr, log, actionAuthor, playContext) {
    var lineTextList = makeCompleteSeqFromChordSeqsAndFormat(chordSeqsAndFormat)

    lineTextObject = new LineTextObject(lineTextList, inputText.getDurations(),
                                        inputText.getFocusCellIdx())

    var queryObject = new QueryObject(lineTextObject, author, 'playSubseq', actionAuthor,
                                      true, false, idStr, i, inputText.val())

    queryObject.chordSeqsAndFormat = chordSeqsAndFormat
    queryObject.log = log
    console.log('chordSeqsAndFormat', chordSeqsAndFormat)
//    socket.emit("playSubseq", chordSeqsAndFormat, 'dont_log', i, idStr, queryObject, log)
    socket.emit("playSubseq", queryObject, log, playContext)
}

// associated with fix text
function makePlayButton(parent, text, author, i, idStr, doAppend) {
    var btn = $('<button>').addClass('btn btn-xs').text('Play')
    if (doAppend) {btn.appendTo(parent)};
    btn.click(function(e) {
        console.log('play text button clicked', text)
        updateSong(text, true, author, idStr, i)
//        socket.emit("playSeq", text, 0, author, i, idStr)
    });
    return btn
}

function updateSong(text, play, author, idStr, i) {
    console.log('updating Song')
    lineTextObject = new LineTextObject(text, inputText.getDurations(),
                                        inputText.getFocusCellIdx())

    playSeq(lineTextObject, author, 'play', play, false, idStr, i)
}


function makePlayChangeButton(parent, text, chordSeqsAndFormat, author, i, idStr, playContext) {
//    var btn = $('<button>').addClass('btn btn-mini').text('Play bold').css('border-radius', '8px').appendTo(parent);
    if (playContext) {
        playLabel = 'Play context'
    }
    else {
        playLabel = 'Play'
    }
    //btn-mini
    var btn = $('<button>').addClass('btn btn-xs').text(playLabel).css('border-radius', '8px').appendTo(parent);
    btn.click(function(e) {
        console.log('playSubseq button clicked', i)

        emitPlaySubseq(text, chordSeqsAndFormat, author, i, idStr, true, 'user', playContext)
//        socket.emit("playSubseq", chordSeqsAndFormat, author, i, idStr)
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
//// display caret pos
//inputText.bind("mouseup",
//    function() {
//        caretPos = $(this).caret().start
//        posInt.text(caretPos)
//});




//// display the past
//inputText.keyup(function(e) {
//    var len = inputHistory.length
//    var value = this.value
//    console.log('keyup, this.value', this.value)
//    //caretPosP = caretPos
//    caretPos = inputText.caret().start
//    posInt.text(caretPos)
//    console.log('keyup', e.which, value, this.value[caretPos-1])
//
//    socket.emit("generateAlternatives", value, caretPos)
//
//    // detect shift enter
//    if (e.which==13 && event.shiftKey) {
//        console.log('shift enter')
//        socket.emit("playSeq", this.value, inputText.caret().start)
//    }
//    else if (e.which==40 && suggestInd < suggestList.length - 1) {
//        console.log('suggest', suggestInd)
//        $('#suggest'+suggestInd).css('background-color', 'white')
//        suggestInd += 1
//        $('#suggest'+suggestInd).css('background-color', '#eee')
//    }
//    else if (e.which==38 && suggestInd > -1) {
//        console.log('suggest', suggestInd)
//        $('#suggest'+suggestInd).css('background-color', 'white')
//        suggestInd -= 1
//        $('#suggest'+suggestInd).css('background-color', '#eee')
//
//    }
//    // arrow left, arrow right(39), back, del, shift(16)
//    else if (value[caretPos-1] == ' ' && e.which!=37 && e.which!=8 && e.which!=46 && e.which!=16) {
//        console.log('not shift enter')
//        socket.emit("textChange", value, caretPos, "edit", "user", undefined)//, caretPosP)
//    };
//});

//
//function appendText(text) {
//    if (inputText.val().length == 0) {
//        inputText.val( text )
//    } else {
//        inputText.val( inputText.val() + " " + text )
//    }
//    socket.emit("generateAlternatives", inputText.val(), inputText.val().length-1)
//};

