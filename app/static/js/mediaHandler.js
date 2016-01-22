
var firstUpdateCellColor = false


var videoTopPanel = $('#videoTop')
var highlightColor = getColorForFocus()
if (!demo) {
    var stepOne = $('<'+instructionStepSize+'>').text('Step 1: ').css('color', highlightColor).appendTo(videoTopPanel)
//var videoText = "Check out the video. Beware that it loops!"
    var videoText = "Check out this image!"
    $('<small>').text(videoText).appendTo(stepOne)
//$('<'+instructionTextSize+'>').text(videoText).appendTo(stepOne)
} else {
    var stepOne = $('<h1>').text('ChordRipple ').css('color', highlightColor).css('margin-left', '-2px').appendTo(videoTopPanel)
    $('<br>').appendTo(videoTopPanel)

    // Transition (typical)
    makeToggleButton('transitionMode', 'Typical', videoTopPanel, false)

    // Similarity (atypical, Chord2Vec)
    makeToggleButton('simMode', 'Similar', videoTopPanel, true)

    makeToggleButton('ripple', 'Ripple', videoTopPanel, true)
    

    // add click to get default sequence
    var defaultSeq = $('<button>').css('margin-left', '400px').addClass('btn btn-default btn-mini').text('Default seq').appendTo(videoTopPanel)
    defaultSeq.click(function(e) {
                    console.log('...defaultSeq clicked')
                    socket.emit("defaultSeq")
                    })
    
}

function makeToggleButton(emitTag, text, parent, indent) {
    var textSpan = $('<span>').text(text+': ').appendTo(parent)
    if (indent) {
        // a hack to space
        textSpan.css('margin-left', '65px')
    }

    var toggleDiv = $('<div>').addClass('toggle-button').css('position', 'absolute').appendTo(parent)

    var toggleButton = $('<button>').css('margin-right', '10px').appendTo(toggleDiv)

    toggleDiv.click(function(e) {
        if (toggleDiv.hasClass('toggle-button-selected')) {
            console.log(emitTag + 'was ON')
            socket.emit(emitTag, false)
        } else {
            console.log(emitTag + 'was OFF')
            socket.emit(emitTag, true)
        }
    })

}



// test embed video
videoIFrame = $('iframe.embedly-embed')[0]

var videoPlayer = new playerjs.Player(videoIFrame);


videoPlayer.on('ready', function(){
    // don't autoplay
    videoPlayer.pause()

    videoPlayer.on('play', function(){
        console.log('--- Video on play ---');
        console.log('simultaneouslyPlay', simultaneouslyPlaybackRequested)
        // TODO: is it fired after pause?
        if (simultaneouslyPlaybackRequested) {
            startPlaybackCallback()
        } else {
            startVideoPlaybackCallback()
        }
    });

    videoPlayer.on('pause', function(){
        console.log('--- Video on pause ---');
        console.log('simultaneouslyPlay', simultaneouslyPlaybackRequested)
        // TODO: does this fire upon buffering? or is it stop?
        if (simultaneouslyPlaybackRequested) {
            stopPlaybackCallback(false)
        } else {
            stopPauseVideoPlaybackCallback(false)
        }
    });


    videoPlayer.on('error', function(){
        console.log('--- VIDEO ERROR occurred---');
        stopPlaybackCallback(true)
    });

    videoPlayer.on('ended', function(){
        console.log('....video playing ended, completely one loop?...')
//        stopPauseSongPlayback(true)
//        stopPlaybackCallback(true)

    });

    videoPlayer.getDuration(function(duration){
        console.log(duration);
    });

    if (videoPlayer.supports('method', 'mute')){
        // mute the sound of the video by default
        videoPlayer.mute();
    }



    // the original video player callback
    //  videoPlayer.on('timeupdate', function(data){
    //    console.log('timeupdate', data)
    //    updateVideoSlider(data.seconds)
    //  });
});


// can't get glyphicon to show yet...
//<button type="button" class="btn btn-default" aria-label="Left Align">
//  <span class="glyphicon glyphicon-align-left" aria-hidden="true"></span>
// </button>



parent = $('#playbackControls')

// TODO: later can add bookmarks / chord change times
//var seekThreeSecond = $('<button>').addClass('btn btn-default').text('4.0s').appendTo(parent)
//seekThreeSecond.click(function(e) {
//   console.log('seek 4.0 seconds clicked')
//   videoPlayer.pause();
//   videoPlayer.setCurrentTime(4.0);
//
//
//   videoPlayer.getCurrentTime(function(value){
//   console.log('getCurrentTime:', value);
//});
//});
//
//
//var seekOneSecond = $('<button>').addClass('btn btn-default').text('6.3s').appendTo(parent)
//seekOneSecond.click(function(e) {
//   console.log('seek 6.3 seconds clicked')
//   videoPlayer.pause();
//   videoPlayer.setCurrentTime(6.3);
//
//
//   videoPlayer.getCurrentTime(function(value){
//   console.log('getCurrentTime:', value);
//});
//});


// ============================
// ==== CONTROLS FOR VIDEO ====
// ============================
$('<br>').appendTo(parent)

// adding video timeline
var videoTimeLinePanel = $('<small>').text('Current time (Video): ').appendTo(parent)
videoTimeLinePanel.css('margin', '2px')
var initialTime = 0.00
//$('<div>').attr('id', 'slider').addClass('.slider .slider-horizontal').appendTo(parent)
var videoSliderSpan = $('<span>').text(initialTime).appendTo(videoTimeLinePanel)
var videoSlider = $('<div>').slider({min:0.00, step:0.01, value:initialTime, max:6.50}).addClass('slider slider-horizontal')
//videoSlider.css({'width': '40%'}).appendTo(parent)
//videoSlider.css({'width': WIDTH+'px'}).appendTo(parent)
videoSlider.appendTo(parent)
var videoCurrentPlayTime = undefined
var videoLoopCount = 0
var videoResetSuccess = true

var justSeeked = false

// adding video buttons
var playVideoButton = $('<button>').addClass('btn btn-mini').text('Play').appendTo(parent)
playVideoButton.css("margin", MARGIN)
//var playVideoButtonSpan = $('<span>').addClass('glyphicon glyphicon-align-left').attr('aria-hidden', "true").appendTo(playVideoButton)
// click callback in sequencer init

// for Vine, there's no stop only pause?
var pauseVideoButton = $('<button>').addClass('btn btn-mini').text('Pause').appendTo(parent)
pauseVideoButton.css("margin", MARGIN)
//var pauseVideoButtonSpan = $('<span>').addClass('glyphicon glyphicon-align-left').attr('aria-hidden', "true").appendTo(pauseVideoButton)

var stopVideoButton = $('<button>').addClass('btn btn-mini').text('Stop').appendTo(parent)
stopVideoButton.css("margin", MARGIN)

// initialization
var VIDEO_COMPONENT_ID = 0
var CHORD_COMPONENT_ID = 2
var COMPONENT_IDS = [VIDEO_COMPONENT_ID, CHORD_COMPONENT_ID]
var component_list = []
for (var i=0; i<COMPONENT_IDS.length; i++) {
    var component = new ComponentState(COMPONENT_IDS[i], false, 0.0, ANIMATION_DELTA_THRESHOLD)
    component_list.push(component)
}
var components = new Components(component_list, undefined)


// init the Midi related objects
var sequencer = window.sequencer
var song  = undefined // what is actively being played back
var song_main = undefined  // the user main sequence
var velocity = 110  // loudness
var instrumentName = 'piano'
var path = '../'
var notesRemainingInLoop = []
var notes = []

sequencer.addAssetPack({url: path + '/soundfont/asset_pack_basic.json'}, init);

function init(){
//    notes = makeNotes()
//    song = makeSong(notes, sequencer)

}

// set up video GUIs and callbacks
playVideoButton.click(function(e) {
    console.log('playVideoButton clicked')
    startVideoPlaybackCallback()
});

pauseVideoButton.click(function(e) {
    console.log('pauseVideoButton clicked')
    stopPauseVideoPlaybackCallback(false);
});

stopVideoButton.click(function(e) {
    console.log('stopVideoButton clicked')
    console.log('simultaneouslyPlay', simultaneouslyPlaybackRequested)
    // TODO: does this fire upon buffering? or is it stop?
    if (simultaneouslyPlaybackRequested) {
        stopPauseVideoPlaybackCallback(true)
    } else {
        stopPlaybackCallback(true)
    }

//    stopPauseVideoPlaybackCallback(true);
});

// manually moved by user...
videoSlider.slider({
    slide: function(event, ui){
        displayValue = ui.value.toFixed(2)
        videoSliderSpan.text(displayValue);
        seekVideoSliderHandlerDebounced(ui.value)
}});

moveVideoSliderToCurrentPlayTime = function (){
//    console.log('moveVideoSliderToCurrentPlayTime, videoCurrentPlayTime:', videoCurrentPlayTime)


    videoPlayer.getCurrentTime(function(value){
        component = components.getComponent(VIDEO_COMPONENT_ID)

//        console.log('video getCurrentTime', value)
//        console.log('getCurrentTime', value, 'videoResetSuccess', videoResetSuccess)
//        console.log('component resetThreshold', component.resetThreshold())

//        if (value < component.resetThreshold()) {
//            videoResetSuccess = true
//        }
//        console.log('videoResetSuccess', videoResetSuccess)

        videoCurrentPlayTime = value % videoDuration
//        console.log('video gettime:', videoCurrentPlayTime)

//            numLoops = Math.floor(value / videoDuration)
//            // if it looped around, then repopulate the notes
//            console.log("num_loops, videoLoopCount", numLoops, videoLoopCount)
//            if (!justSeeked && numLoops > videoLoopCount) {
//                console.log("==================")
//                console.log('...refilling notes')
//                console.log("==================")
//                notesRemainingInLoop = notes.slice(0)
//                videoLoopCount = numLoops
//            }
        // update interface
        videoSliderSpan.text(videoCurrentPlayTime.toFixed(2));
        videoSlider.slider('value', videoCurrentPlayTime)
    });
//        console.log('after getCurrentTime', videoCurrentPlayTime)
}

function moveSliderSetVideoTime(time){
    videoSliderSpan.text(time.toFixed(2));
    videoSlider.slider('value', time)
    videoPlayer.setCurrentTime(time)
}

function seekVideoSliderHandler(time) {
    console.log("**********")
    console.log("seeked", time)
    console.log("**********")
    seekPlayback(time, VIDEO_COMPONENT_ID)
}

// when triggered by user?
// TODO: not sure this is doing anything yet
seekVideoSliderHandlerDebounced = _.debounce(function(time){seekVideoSliderHandler(time)});


console.log('...init sequencer')
console.log('sequencer PPQ', sequencer.defaultPPQ)



// ============================
// ==== CONTROLS FOR MIDI =====
// ============================

var WANT_CONTROLS_FOR_MIDI = false
if (WANT_CONTROLS_FOR_MIDI) {
$('<hr>').appendTo(parent)

var midiTimeLinePanel = $('<p>').text('Current time (Chords): ').appendTo(parent)
midiTimeLinePanel.css('margin', '2px')
var initialTime = 0.00
//$('<div>').attr('id', 'slider').addClass('.slider .slider-horizontal').appendTo(parent)
var midiSliderSpan = $('<span>').text(initialTime).appendTo(midiTimeLinePanel)
var midiSlider = $('<div>').slider({min:0.00, step:0.01, value:0, max:videoDuration})
midiSlider.addClass('slider slider-horizontal').appendTo(parent)
//.css({'width': WIDTH+'px'}).appendTo(parent)
//    var midiSliderWidth

midiSlider.slider({
    slide: function(event, ui){
        displayValue = ui.value.toFixed(2)
        midiSliderSpan.text(displayValue);
        midiSliderHandlerDebounced(ui.value)
}});

function updateSongWidgetsToCurrentPlayTime(currentMidiPlayTime){
//        seconds = computeCurrentMidiSliderPosition(currentMidiPlayTime)
    if (currentMidiPlayTime === undefined) {
        currentMidiPlayTime = getMidiPlayTime()
    }
//    if (currentMidiPlayTime > videoDuration) {
//        console.log(',,updateSongWidgetsToCurrentPlayTime', currentMidiPlayTime, videoDuration)
//        stopPauseVideoPlaybackCallback(true)
//    }

    if (WANT_CONTROLS_FOR_MIDI && !playingSubseq) {
        midiSliderSpan.text(currentMidiPlayTime.toFixed(2))
        midiSlider.slider('value', currentMidiPlayTime)
    }

    var offset = 0
    var startIdx = undefined
//    console.log('...getActiveInputText() === inputText', getActiveInputText() === inputText)
    var playedRecommendation = queryState.actionKind.indexOf('play') >= 0 && queryState.actionAuthor == 'user'
                                && queryState.author == 'machine'
    var usedRecommendation = queryState.actionKind == 'use' && queryState.author == 'machine'
//    console.log('...playedRecommendation', playedRecommendation, queryState.actionAuthor, queryState.actionKind)

    // user played his/her own sequence or recommended seq
    // should update playhead, since not playing singleton by moving focus
    // and not machine moving focus
    var userPlayedSeq = queryState.actionAuthor == 'user' && queryState.actionKind.indexOf('play') >= 0
//    console.log('...author', queryState.author, 'actionKind', queryState.actionKind,
//                 'actionAuthor', queryState.actionAuthor)

    if ( usedRecommendation || playedRecommendation) {
        var chordSeqsAndFormat = queryState.chordSeqsAndFormat
        startIdx = getFirstChangedChordIdx(chordSeqsAndFormat)
        offset = videoScore.getTimings()[startIdx]
//        console.log('...need to compute offset', startIdx)
    } else if (!this.focusMovedByMachine && !userPlayedSeq && getActiveInputText() === inputText) {
        // user moved focus
        currentFocusIdx = inputText.getFocusCellIdx(true)
//        console.log('UpdateSongW, currentFocusIdx', currentFocusIdx)
        offset = videoScore.getTimings()[currentFocusIdx]
//        if (currentFocusIdx != undefined) {
//            offset = videoScore.getTimings()[currentFocusIdx]
//        }

    }

//    console.log('...updatePlayhead offset', offset)
//    console.log('currentMidiPlayTime', currentMidiPlayTime, currentMidiPlayTime+offset)

    var activeInputText = getActiveInputText()
//    console.log('...activeInputText', activeInputText)
    if (activeInputText == undefined) {
        console.log('...WARMING: activeInputText is undefined')
        return currentMidiPlayTime
    }

    var firstUpdateForUserFocusedCell = userFocusedCell && firstUpdateCellColor
//    console.log('userFocusedCell', userFocusedCell, 'firstUpdateCellColor', firstUpdateCellColor)
//    console.log('queryState.author', queryState.author, 'queryState.actionKind', queryState.actionKind)
//    console.log('userPlayedSeq', userPlayedSeq)

    var machinePlayedUse = queryState.actionKind == 'use'

    // here is where the playhead color gets updated
    if (!userFocusedCell || firstUpdateForUserFocusedCell || userPlayedSeq || machinePlayedUse) {
        activeInputText.updatePlaybackColorCell(currentMidiPlayTime + offset)
        firstUpdateCellColor = false
    } else {
        console.log('...not updating color')
    }
    return currentMidiPlayTime
}

function getFirstChangedChordIdx(chordSeqsAndFormat) {
    var changeIdx = undefined
    for (var i=0; i<chordSeqsAndFormat.length; i++) {
        if (chordSeqsAndFormat[i][1]) {
            changeIdx = i
            break
        }
    }
    return changeIdx
}

function getActiveInputText() {
    var sideEffectInputFocus
    var useActionKind = queryState.actionKind == 'use' || queryState.actionKind == 'inputFocus'
    var playingUserSeq = queryState.author == 'user'
//    console.log('..getActiveInputText, playingSubseq', playingSubseq, queryState)

    var playSuggestion = queryState.panelId != undefined && queryState.itemIdx != undefined
//    console.log('useActionKind', useActionKind, 'queryState.panelId', queryState.panelId,
//                queryState.itemIdx)

    var activeInputText = undefined
    if (playingUserSeq || useActionKind) {
//        console.log('inputText')
        activeInputText = inputText
    } else if (queryState.author == 'input') {
        // TODO: need to make saved entries into InputTextCellSequences
        activeInputText = undefined

    } else {
        var itemIdx = queryState.itemIdx
        if (queryState.panelId.indexOf("above") > -1) {
//            console.log('above suggestions')
            activeInputText = suggestionsAboveInputText[itemIdx]
        } else {
//            console.log('below suggestions')
            activeInputText = suggestionsInputText[itemIdx]
        }
    }
    return activeInputText
}


function midiSliderHandler(time) {
    bufferMidi(time)
    seekVideo(time)
}

midiSliderHandlerDebounced = _.debounce(function(time){
    midiSliderHandler(time)
});


var startPlay = $('<button>').addClass('btn btn-mini').text('Play').appendTo(parent)
startPlay.css('margin', MARGIN)
startPlay.click(function(e) {
   console.log('start play note hopefully!')
   if (!song_main === undefined) {
       song = song_main
   }
   startSongPlaybackCallback()
});

var pausePlay = $('<button>').addClass('btn btn-mini').text('Pause').appendTo(parent)
pausePlay.css('margin', MARGIN)
pausePlay.click(function(e) {
   console.log('pause play note hopefully!')
   stopPauseSongPlaybackCallback(false)
});

var stopPlay = $('<button>').addClass('btn btn-mini').text('Stop').appendTo(parent)
stopPlay.css('margin', MARGIN)
stopPlay.click(function(e) {
   console.log('stop play note hopefully!')
   stopPauseSongPlaybackCallback(true)
});

//var remakeSong = $('<button>').addClass('btn btn-mini').text('Make song').appendTo(parent)
//remakeSong.css('margin', MARGIN)
//remakeSong.click(function(e) {
//   console.log('remakeSong hopefully!')
//   makeSongWithVideoTiming()
//});

}


// ============================
// CONTROLS BOTH VIDEO AND MIDI
// ============================

$('<hr>').appendTo(parent)


var stepThree = $('<'+instructionStepSize+'>').text('Step 3: ').css('color', highlightColor).appendTo(parent)
var playbackBothText = "See / hear how well they go together.  Note the sync for this alpha version is still very loose."
$('<small>').text(playbackBothText).appendTo(stepThree)

// "So we're just going for an overall fit."
//$('<'+instructionTextSize+'>').text(playbackBothText).show().appendTo(parent)


var startAnimating = $('<button>').addClass('btn btn-mini').text('Play both').appendTo(parent)
startAnimating.css("margin", MARGIN)
startAnimating.click(function(e) {
   console.log('Start repaint callback!', components.onPlay())

   // if not already in play state, false is for not called after seek
   // only checked once
//   if (!song_main === undefined) {
//       song = song_main
//   }
   // TODO: do I need to check some condition for stop?
//   stopPlaybackCallback(true)
   simultaneouslyPlaybackRequested = true
   // this emits playSeq to synth most updated seq before play
   playSeq(inputText, 'user', 'play both', false, false)//, idStr, i)


});

//function makeSongWithVideoTiming() {
//// need to make a version of the score that respects the absoluate timings
//   if (socket != undefined) {
//        // TODO: use some other way to guarantee the ratio comes to be 1
//        var speed = 60.0
//        socket.emit('setPlaybackSpeed', speed)
//        // just want to send back song...
//        var queryObject = new QueryObject(inputText, 'dont_log', 'artificialPlay', 'machine', false, false)
//        console.log('inputTextPlayButton', queryObject)
//        socket.emit("playSeq", queryObject)
//   }
//}

var pauseAnimating = $('<button>').addClass('btn btn-mini').text('Pause both').appendTo(parent)
pauseAnimating.css("margin", MARGIN)
pauseAnimating.click(function(e) {
   console.log('pause repaint callback!', components.onPlay())
   stopPlaybackCallback(false)
});

var stopAnimating = $('<button>').addClass('btn btn-mini').text('Stop both').appendTo(parent)
stopAnimating.css("margin", MARGIN)
stopAnimating.click(function(e) {
   console.log('Stop repaint callback!', components.onPlay())
   simultaneouslyPlaybackRequested = false
   stopPlaybackCallback(true)
});

$('<hr>').appendTo(parent)


function getMidiPlayPercentage() {
    return song.percentage
}

function getMidiPlayTime() {
//        console.log('song.ticks', song.ticks, 'song.bpm', song.bpm)
    if (song !== undefined) {
        var numQuarterNotes = song.ticks / song.PPQ
        var seconds = numQuarterNotes * 60 / song.bpm
    //        console.log('song.PPQ', song.PPQ, 'numQuarterNotes', numQuarterNotes)
    //        console.log('seconds', seconds)
        return seconds
    }
    return 0
}

function bufferMidi() {
//    console.log('...bufferMidi, videoCurrentPlayTime', videoCurrentPlayTime)
    if (videoCurrentPlayTime === undefined) {
        return videoCurrentPlayTime
    }
    var songDuration = ticksToSecond(song.durationTicks)
//    console.log('songDuration', songDuration)
//    if (videoCurrentPlayTime > songDuration-ANIMATION_DELTA_THRESHOLD) {
    if (videoCurrentPlayTime > songDuration) {
        console.log('--- Song stopped --- exceeded duration', songDuration)
        song.stop()
        return songDuration
    }


//    console.log('bufferMidi', song.playing)
    if (!song.playing) {
        song.play()
    }

    var midiPlayTime = getMidiPlayTime()
    var time_delta = midiPlayTime - videoCurrentPlayTime

//    console.log('bufferMidi', videoCurrentPlayTime, midiPlayTime, time_delta)

    var time_delta_abs = Math.abs(time_delta)
    if (time_delta_abs < videoMidiOffThreshold) {
//        console.log('...time delta small, paused?', song.paused, time_delta)
        if (song.paused) {
            song.play()
            console.log('--- Song play ---')
        }
        // tempo might have been changed before, if already close enough in sync
        // adjust the tempo back to normal
//            console.log('close enough to threshold, current bpm', song.bpm)
//            if (song.bpm != bpm && song.bpm != NaN) {
//                console.log('......setting song tempo', setTempo)
//                song.setTempo(bpm)
//            }

//        console.log('close enough dont need to adjust', song.bpm)
//            console.log('after adjusting bpm', song.bpm)
    } else {
        // if video lagged too much behind, then pause
        // but need midi to loop around, so if the lag happens at the very end
        // TODO: hack breaks if lag happens at the very end...
        console.log('...time delta, paused?', time_delta, song.paused)
        if (time_delta > 1 && time_delta < 1.5 && midiPlayTime < 6.4) {
            console.log('--- pausing song ---')
            if (!song.paused) { song.pause() }
        } else {
            song.setPlayhead('millis', videoCurrentPlayTime*1000)
            console.log('--- resetting song playhead ---', song.ticks)
            if (song.paused) { song.play() }
            midiPlayTime = getMidiPlayTime()
        }
    }
//        else { // if too off then adjust playback tempo
//            var tempoAdjustRatio = compute_makeup_tempo_ratio(time_delta)
//            console.log('tempoAdjustRatio', tempoAdjustRatio)
//            song.setTempo(tempoAdjustRatio*bpm)
//            console.log('after adjusting tempo, bpm', song.bpm)
//        }
    return midiPlayTime
}



function compute_makeup_tempo_ratio(time_delta) {
    // adjust playback speed
    // what are the assumptions
    // 1) want to not speed up too much, <1.3
    // 2) want to not lag behind for too long, vice versa, 0.5s
    // if video is ahead, then need to fill up that empty space while still catching up
    var beat_dist = time_delta * bpm
    var TIME_SLACK = 0.5
    // TODO: if we want a limit to how much it should speed up
    var tempoRatio = TIME_SLACK / (beat_dist + TIME_SLACK)
    return tempoRatio
}

// setting up the repaint...
var requestAnimationFrame = window.requestAnimationFrame
var videoMidiOffThreshold = 0.2  // allow them to be 100ms apart

// }  //originally init()


// =======================================
// ======= animation callbacks ===========
// =======================================

var animationCallbackRequest = true

function updateVideoHandler() {
    moveVideoSliderToCurrentPlayTime()
    animationCallbackRequest = requestAnimationFrame(updateVideoHandler)
}

function updateSongHandler() {
    currentMidiPlayTime = updateSongWidgetsToCurrentPlayTime()

//    console.log('updateSongHandler, playbackStopTime, currentMidiPlayTime', playbackStopTime, currentMidiPlayTime)
    if (song != undefined) {
//        console.log('updateSongHandler, endOfSong, looping', song.endOfSong, looping)
        var endOfSong = song.endOfSong && !looping
    }
    else {
        var endOfSong = true
    }
    var beyondStopTime = playbackStopTime - DONT_INCLUDE_ONSET_OF_NEXT_NOTE_EPSILON < currentMidiPlayTime

    if (song == undefined || endOfSong || beyondStopTime) {
        console.log('updateSongHandler stopping song', undefined, endOfSong, beyondStopTime)
        stopPauseSongPlaybackCallback(true)
        // moved into stopPauseSongPlaybackCallback
//        var activeInputText = getActiveInputText()
//        if (activeInputText != undefined) {
//            activeInputText.resetCellColorExceptIdx()
//        }
    } else {
        animationCallbackRequest = requestAnimationFrame(updateSongHandler)
    }
}

function updateSyncBetweenVideoAndMidiHandler() {
//    console.log('...updateSyncBetweenVideoAndMidiHandler')
    moveVideoSliderToCurrentPlayTime()

    currentMidiPlayTime = bufferMidi()
    updateSongWidgetsToCurrentPlayTime(currentMidiPlayTime)
    animationCallbackRequest = requestAnimationFrame(updateSyncBetweenVideoAndMidiHandler)
}


function preparePlaySong(loop, offset) {
    firstUpdateCellColor = true
    // the offset comes in as seconds
    if (offset == undefined) {
        offset = 0
    }
   if (song.playing) {
       song.stop()
   }
   console.log('...in pause mode', song.paused)
   if (!song.paused || offset > 0) {
       console.log('...setting playhead to', offset)
       console.log('total length of song', ticksToSecond(song.durationTicks))
       song.setPlayhead('millis', offset*1000)
   }
   if (loop) {
       song.setLeftLocator('ticks', secondToTicks(offset))
       song.setRightLocator('ticks', song.durationTicks)
       song.setLoop(true)
   }
}

function playSong(loop, offset) {
    preparePlaySong(loop, offset)
    song.play()
}

function playVideo() {
    videoPlayer.play()
//    videoPlayer.getPaused(function(paused){
//        console.log('--- Video was paused:', paused);
//        if (paused) {
//            videoPlayer.pause()
//        }
//        else {
//            videoPlayer.play()
//        }
//    });

}

function startPlaybackCallback() {
    console.log('startPlayback, components.onPlay()', components.onPlay())
    looping = true
    // if was in playState continue to play
    if (!components.onPlay() || components.onPlayAndSeeked(VIDEO_COMPONENT_ID)) {
        // videoPlayer already set to 0 at stop
        // videoPlayer.setCurrentTime(0)
        playVideo()

        // wait for video to start playing before playing
        preparePlaySong()

        components.setToPlay()
        animationCallbackRequest = requestAnimationFrame(updateSyncBetweenVideoAndMidiHandler)

   }
}

function startVideoPlaybackCallback() {
    playVideo()
    components.setToPlay(VIDEO_COMPONENT_ID)
    animationCallbackRequest = requestAnimationFrame(updateVideoHandler)
}

function startSongPlaybackCallback(loop, offset) {
    console.log('...startSongPlaybackCallback loop', loop, 'song.paused', song.paused)
    // TODO: but previous song has been overwritten already so not sure we can event stop it here
    // TODO: should wrap song in a class so that can control access
    if (!loop && !song.paused && song.playing) {
        console.log('...stopping song playback before start (might have a problem if paused')
        stopPauseSongPlaybackCallback(true)
    }
    playSong(loop, offset)
    components.setToPlay(CHORD_COMPONENT_ID)
    animationCallbackRequest = requestAnimationFrame(updateSongHandler)
}

function resetSong() {
    if (song != undefined) {
        if (!song.paused && song.playing) {
            console.log('...resetSong: previous song is playing and not paused', song.playing)
            stopPauseSongPlaybackCallback(true)
            var activeInputText = getActiveInputText()
            activeInputText.resetCellColorExceptIdx()
        } else if (song.paused) {
            console.log('song was paused so not stopping')
        }
    }
}

function startChordsPlaybackCallback(loop) {
    playSong(loop)
//    components.setToPlay(CHORD_COMPONENT_ID)
//    animationCallbackRequest = requestAnimationFrame(updateSongHandler)
}






function seekPlayback(seekTime, componentId) {
    console.log('--- seekPlayback seekTime', seekTime, 'onPlay', components.onPlay())
    components.getComponent(componentId).seekTime = seekTime
    components.lastSeekedComponent = componentId
    // equivalent to stop and then start
    // but then need to start at different place and append the notesTooLate
    // perhaps keep the last one...but could be from long time ago so...

    // does not reset video
    // stopPlayback(false)
    seekVideo(seekTime)

    // continue to play if was in playing mode
    if (components.onPlay()) {
        startPlaybackCallback(true)
    }
}

function seekVideo(time){
   videoPlayer.setCurrentTime(time)
   videoComponent = components.getComponent(VIDEO_COMPONENT_ID)
   videoComponent.seekTime = time
   // this for discarding the earlier notes before seek time
   justSeeked = true
}


// ======== stop the callbacks and the playbacks ========
function stopPlaybackCallback(stop) {
//    console.log('stopPlayback, onPlay()', components.onPlay())
    console.log('.stopPlaybackCallback')
//        if (components.onPlay()) {
    looping = false
    stopPauseVideoSongPlayback(stop)
    window.cancelAnimationFrame(animationCallbackRequest)
    components.setToStop()
//        }
}

function stopPauseVideoPlaybackCallback(stop) {
    console.log('stopVideoPlayback, onPlay()', components.onPlay())
    // want to be able to stop when on pause
    // if (components.onPlay(VIDEO_COMPONENT_ID)) {
    stopPauseVideoPlayback(stop)
    window.cancelAnimationFrame(animationCallbackRequest)
    components.setToStop(VIDEO_COMPONENT_ID)
//        }
}

 function stopPauseSongPlaybackCallback(stop) {
    console.log('stopVideoPlayback, onPlay()', components.onPlay())
    // want to be able to stop when on pause
    // if (components.onPlay(CHORD_COMPONENT_ID)) {
    stopPauseSongPlayback(stop)
    window.cancelAnimationFrame(animationCallbackRequest)
    components.setToStop(CHORD_COMPONENT_ID)
//        }
 }

// ======== just stop the playback ========
function stopPauseVideoSongPlayback(stop){
    console.log('stopPauseVideoSongPlayback', stop)
    stopPauseVideoPlayback(stop)
    stopPauseSongPlayback(stop)
}

function stopPauseVideoPlayback(stop){
    console.log('---stopPauseVideoPlayback', stop)
    videoPlayer.pause()
    if (stop) {
        moveSliderSetVideoTime(0)
    }
}

function stopPauseSongPlayback(stop) {
    console.log('---stopPauseSongPlayback, stop', stop)
    // end of user focus cell trigger playback if that was the case, otherwise doesn't matter
    // can't set to false here because next one might be userFocusedCell too
//    userFocusedCell = false

    if (song == undefined) {
        return
    }
    if (stop) {
        song.stop()
        console.log('--- song stopped ---', song.ticks)
//        song.setPlayhead('millis', 0)

    } else if (!song.paused && !song.stopped) {
        // pause is actually a toggle between play and pause
        song.pause()
        console.log('--- song paused ---')
    }

    var activeInputText = getActiveInputText()
    if (activeInputText != undefined && stop) {
        activeInputText.resetCellColorExceptIdx()
    }

}


