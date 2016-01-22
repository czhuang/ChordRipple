

var previousFocusIdx = undefined
//var timings = [0.0, 0.8, 1.5, 3.27, 3.9, 5.3, 5.78, 6.5]
// var timings = [0.0, 0.8, 1.5, 3.3, 3.9, 5.3, 5.78]
var timings = [ 0.      ,  0.65    ,  1.21875 ,  2.656875,  3.16875 ,  4.30625 ,
                4.69625 ,  5.28125 ]
// if start from second chord, 0.15, 0.7, 2.5
var videoScore = new VideoScore(timings)
var ANTICIPATE_AMOUNT = 0.0

var tempScrollTop = undefined
var userFocusedCell = undefined


function VideoScore(videoTimings) {
    this.makeEvent = function(videoTime, chordInd) {
        event = {}
        event.videoTime = videoTime
        event.chordInd = chordInd
        return event
    }

    this.initSyncEvents = function(videoTimings) {
        if (videoTimings === undefined) {
            return []
        } else {
            var syncEvents = []
            for (var i=0; i < videoTimings.length; i++) {
                syncEvents.push(this.makeEvent(videoTimings[i], i))
            }
            return syncEvents
        }
    }
    this.syncEvents = this.initSyncEvents(videoTimings)

    this.addEvent = function(videoTime, chordInd) {
        this.syncEvents.push(this.makeEvent(videoTime, chordInd))
    }

    this.getTimings = function() {
        var timings = []
        for (var i=0; i<this.syncEvents.length; i++) {
            var time = this.syncEvents[i].videoTime
            timings.push(time)
        }
        return timings
    }

    this.length = function() { return this.syncEvents.length }

    this.getEventIdxFromTime = function(currentTime) {
        var idx = 0
        for (var i=0; i<this.syncEvents.length; i++) {
            var time = this.syncEvents[i].videoTime
//            if (currentTime > time - ANTICIPATE_AMOUNT) {
//            console.log('getEventIdxFromTime, currentTime', currentTime)
//            console.log('check time', time, time + MIDI_START_SLACK)
            if (currentTime >= time){ // + MIDI_START_SLACK) {
                idx = i
            }
//            console.log(time, currentTime, i, idx)
        }
//        console.log('timings', this.getTimings())
//        console.log('currentTime, idx', currentTime, idx)
        return idx
    }

    this.getTimeFromEventIdx = function(idx) {
        if (idx >= this.syncEvents.length) {
            idx = this.syncEvents.length - 1
        }
        return this.getTimings()[idx]
    }
}

function QueryObject(inputText, author, actionKind, actionAuthor, play, loop,
                     panelId, itemIdx, userText) {

    if (inputText instanceof InputTextCellSequence ) {
        this.text = inputText.val()
        this.seq = inputText.getTextList()
        this.activeIdx = inputText.getFocusCellIdx()
        this.durations = inputText.getDurations()
    } else {
        this.text = inputText.text
        this.seq = inputText.seq
        this.activeIdx = inputText.activeIdx
        this.durations = inputText.durations
    }
    this.author = author
    this.actionKind = actionKind
    this.actionAuthor = actionAuthor
    this.play = play
    this.loop = loop

    // only when author is computer, suggestions info
    this.panelId = panelId
    this.itemIdx = itemIdx

    // gives the context of what user had before substitution is used
    this.originalText = userText

}

function LineTextObject(inputText, durations, activeIdx) {

    this.setText = function(text) {
        if (typeof(text) == 'string') {
            return text
        }
        else {
            var lineText = ''
            for (var i=0; i<text.length; i++) {
                lineText += text[i] + ' '
            }
            return lineText.trim()
        }
    }

    this.text = this.setText(inputText)

    this.getTextList = function(text) {
        if (typeof(text) == 'string') {
            return text.trim().split(' ')
        } else {
            // assumes that it is a list
            return text
        }
    }
    this.seq = this.getTextList(inputText)

//    this.selectDurations = function(durations) {
//    // TODO: assumes we're starting from the beginning
//        var durs = []
//        for (var i=0; i<this.seq.length; i++) {
//            durs.push(durations[i])
//        }
//        return durs
//        console.log('LineTextObject: durations', durs.length, durs)
//    }

//    this.durations = this.selectDurations(durations)

    this.durations = durations

    this.activeIdx = activeIdx
}

function InputTextCellSequence(grandparentId, cellType, videoScore, socket, width, author,
                               panelId, itemIdx, recType) {
    // the parent should be the immediate parent
    // so that can loop through for example to get which is in focus
    this.parentId = grandparentId + '_child'

    this.makeSpan = function(grandparentId) {
        var grandparent = $('#'+grandparentId)
        var span = $("<span>").attr('id', this.parentId).appendTo(grandparent)
        return span
    }

    this.cellType = cellType

    this.parent = this.makeSpan(grandparentId)
    this.videoScore = videoScore

    this.focusMovedByMachine = false

    this.width = width

    this.author = author
    this.panelId = panelId
    this.itemIdx = itemIdx

    // recommendation type
    this.recType = recType

    this.socket = socket
    this.setSocket = function(socket) {
        this.socket = socket
    }

//    this.editTextNotIndexed = false

//    previousFocusIdx = undefined

    this.getDurations = function() {
        var durations = []
        var startTime = this.videoScore.syncEvents[0].videoTime
        var endTime
        for (var i=0; i<this.videoScore.length(); i++) {
            //TODO: assumes that videoScore is sorted
            if (i==this.videoScore.length()-1) {
                endTime = videoDuration
            }
            else {
                endTime = this.videoScore.syncEvents[i+1].videoTime
            }
            var diffTime = endTime - startTime
            durations.push(diffTime)
            startTime = endTime
        }
        return durations
    }

    this.setupCellsToVideoScore = function() {
        var durations = this.getDurations()
        for (var i=0; i<this.videoScore.length(); i++) {
            var width = durations[i]/videoDuration * this.width
            var cell = this.makeCell(true, i, width)
            this.addCell(cell, i)

        }
    }

    this.makeCell = function(editable, i, width) {
        var id = this.parentId + i
//        var cell = $('<input>').addClass('text').attr('id', id).css({"font-family":"monospace", "font-size":16, "margin-top":8})
        var cell = $('<'+this.cellType+'>').addClass('text').attr('id', id).css({"font-family":"monospace", "font-size":16})

        //cell.css("margin-bottom", '1px')


//      //.width(widthPercent).appendTo(parent);
        if (width === undefined) {
            cell.css({"width":'50px'})
        }
        else {
            cell.css('width', width+'px')
        }
        return cell
    }

    this.addEmptyCells = function(numCell, editable) {
        var cells = this.getCells()
        for (var i=0; i<numCell; i++) {
            var idx = cells.length+i
            var cell = this.makeCell(editable, idx)
            console.log('--- addEmptyCells adding focus, keyup ')
            this.addCell(cell, idx)
        }
    }

    this.addCell = function(cell, idx) {
        cell.appendTo(this.parent)
        this.focusCell(this, idx, cell, this.videoScore, this.socket,
                       'machine')
        this.keyup(this, idx, cell, this.socket, 'machine')
        //this.onClick(this, idx, cell)
    }

    this.getCells = function() {
        // first get a count of child
        // some how the children got through here is of type "[object HTMLInputElement]"
        // but want type "[object Object]"
        var childrenOfTypeText = this.parent.children().filter('.text')
        var numTextBoxes = childrenOfTypeText.length
        var cells = []
        for (var i=0; i<numTextBoxes; i++) {
            var cell = $('#'+this.parentId+i)
            cells.push(cell)
        }
        return cells
    }

    this.getLastNonEmptyCellIdx = function() {
        var cells = this.getCells()
        var lastNonEmptyIdx = cells.length

        for (var i=cells.length-1; i>=0; i--) {
            if (cells[i].val().length != 0) {
                lastNonEmptyIdx = i
                break
            }
        }
        return lastNonEmptyIdx
    }

    this.updatePlaybackColorCell = function(currentTime) {
//        console.log('updatePlaybackColorCell', currentTime)
        var playIdx = this.videoScore.getEventIdxFromTime(currentTime)
        var lastIdx = this.getLastNonEmptyCellIdx()
//        console.log('playIdx, lastIdx', playIdx, lastIdx)
        // if already last idx, don't move
        if (playIdx <= lastIdx) {
            var cells = this.getCells()
            // pinkish color
//            console.log('...updatePlaybackColorCell playIdx', playIdx, 'cells.length', cells.length)
            cells[playIdx].css('background', '#FF5880')
            this.resetCellColorExceptIdx(playIdx)
        }
    }

    this.resetCellColorExceptIdx = function(idx) {
        var cells = this.getCells()
//        console.log('resetCellColorExceptIdx idx', idx)
        for (var i=0; i<cells.length; i++) {
            if (i != idx) {
                // white color
                cells[i].css('background', '#FFFFFF')
            }
        }
    }

    this.resetCellColor = function(idx) {
        var cells = this.getCells()
        for (var i=0; i<cells.length; i++) {
            if (i != idx) {
                // white color
                cells[i].css('background', '#FFFFFF')
            }
        }
    }

    this.getTextList = function() {
        var textList = []
        var cells = this.getCells()
        for (var i=0; i<cells.length; i++) {
            textList.push(cells[i].val().trim())
        }
        return textList
    }

    this.getWidths = function() {
        var cells = this.getCells()
        var widths = []
        for (var i=0; i<cells.length; i++) {
            widths.push(cells[i].outerWidth())
        }
        return widths
    }

    this.val = function(text) {
        if (text === undefined) {
            var text = ''
            var cells = this.getCells()
            for (var i=0; i<cells.length; i++) {
                text += cells[i].val().trim() + ' '
            }
            return text.trim()
        } else {
            text = text.trim().split(' ')

            // not adding cells beyond the video timing
//            var cells = this.getCells()
//            if (text.length >= cells.length) {
//                var diff = text.length - cells.length
////                this.addEmptyCells(diff+1)
//                this.addEmptyCells(diff)
//            }

            cells = this.getCells()
            for (var i=0; i<cells.length; i++) {
                if (i >= text.length) {
                    cells[i].val('')
                } else {
                    cells[i].val(text[i])
                }
            }
        }
    }

    this.setCells = function(textList) {
        cells = this.getCells()
        for (var i=0; i<cells.length; i++) {
            if (i >= textList.length) {
                cells[i].val('')
            } else {
                cells[i].val(textList[i])
            }
        }
    }

    this.moveFocus = function(idx) {
        this.focusMovedByMachine = true
        var cells = this.getCells()
        cells[idx].focus()
    }

    this.updateFocus = function() {
        this.focusMovedByMachine = true
        var cells = this.getCells()
        var firstEmptyCell = undefined
        for (var i=0; i<cells.length; i++) {
            if (cells[i].val().trim().length == 0) {
                firstEmptyCell = i
                break
            }
        }
        // if we were on focus on the last unempty cell,
        // then move to next empty cell
        if (previousFocusIdx+1 == firstEmptyCell) {
            this.moveFocus(firstEmptyCell)
        }
        else {
            this.moveFocus(previousFocusIdx)
        }
        console.log("...updateFocus, previousFocusIdx", previousFocusIdx, 'firstEmptyCell', firstEmptyCell)
    }

    this.getFocusCellIdx = function(returnAlwaysCurrentAlso) {
        var focusIdx = undefined
        var cells = this.getCells()
        for (var i=0; i<cells.length; i++) {
            if (cells[i][0]==document.activeElement) {
                focusIdx = i
                break
            }
        }
        console.log('lineText.getFocusCellIdx: focusIdx', focusIdx)

//        // if it's undefined then keep it as undefined
//        return focusIdx
//        TODO: not sure if there's code dependent on focusIdx being previousFocusIdx
        if (focusIdx == undefined) {
            console.log('previousFocusIdx', previousFocusIdx)
            return previousFocusIdx
//            if (returnAlwaysCurrentAlso) {
//                return focusIdx, previousFocusIdx
//            } else {
//                return previousFocusIdx
//            }
        } else{
            return focusIdx
        }
    }

    this.focus = function() {
        var cells = this.getCells()
        for (var i=0; i<cells.length; i++) {
            this.focusCell(this, i, cells[i], this.videoScore, this.socket)
        }
    }

    this.focusCell = function(parent, ind, cell, videoScore, socket) {
        cell.focus(function(e) {
            previousFocusIdx = ind
            console.log('focusCell, setting previousFocusIdx', previousFocusIdx, 'ind', ind)
            parent.focusCellHandlerDebounced(parent, ind, cell, videoScore, socket)
        })
    }

    this.focusCellHandlerDebounced = _.debounce(
        function(parent, ind, cell, videoScore, socket){
            this.focusCellHandler(parent, ind, cell, videoScore, socket)
    }, 500);


    this.focusCellHandler = function(parent, ind, cell, videoScore, socket) {
        console.log('.....focusCellHandler, machine', this.focusMovedByMachine)
        if (!this.focusMovedByMachine) {
            userFocusedCell = true
        }
        console.log('user', userFocusedCell)
        tempScrollTop = $(window).scrollTop();
        if (videoScore.length() > ind) {
            var seekTime = videoScore.syncEvents[ind].videoTime
//            seekVideo(seekTime)
            moveSliderSetVideoTime(seekTime)
        }

        if (ind==0 && parent.val().length==0) {
            socket.emit("startSeqs")
        } else {

            // also ask for corresponding suggestions
            // need to make query object
            var play = true
            if (looping) {
                play = false
            }
            console.log('looping?', looping, 'want to play', play)
            var queryObject = new QueryObject(parent, parent.author, 'inputFocus', 'user',
                                              play, false)
            var chordSeqsAndFormat = parent.makeChordSeqsAndFormat()
            queryObject.chordSeqsAndFormat = chordSeqsAndFormat
            // distinguish between if the focus was caused programmatically or by the user
            // won't fire text change if it is moved by machine
            console.log('this.focusMovedByMachine', this.focusMovedByMachine)
            if (!this.focusMovedByMachine){ //} && parent.author == 'machine') {

                console.log("focus cell fire textChange", 'chordSeqsAndFormat', chordSeqsAndFormat)
                console.log("query", queryObject)
                socket.emit("textChange", queryObject)
            }
            this.focusMovedByMachine = false
            // don't do anything if focus is moved by machine
            // else focus will be at a position where there be no chords to play
            // or if there is a chord it's not meant to be played
//            else {
//                this.focusMovedByMachine = false
//                // just fire subseq here...
//                queryObject.panelId = parent.panelId
//                queryObject.itemIdx = parent.itemIdx
//                socket.emit("playSubseq", queryObject)
//            }

        }
    }




    this.keyup = function() {
        var cells = this.getCells()
        for (var i=0; i<cells.length; i++) {
            this.keyupCell(this, i, cells[i], this.socket)
        }
    }
    this.keyupCell = function(parent, ind, cell, socket) {
        cell.keyup(function(e) {
            console.log('keyup', e.which)
            parent.keyupHandlerDebounced(parent, e, ind, cell, socket)
        })
    }

    this.keyupHandlerDebounced = _.debounce(
        function(parent, e, ind, cell, socket){
            this.keyupHandler(parent, e, ind, cell, socket)
    }, 300);

    this.keyupHandler = function(parent, e, ind, cell, socket) {
        console.log('keyupHandler actually called')
        this.editTextNotIndexed = true
        tempScrollTop = $(window).scrollTop();
        var queryObject = new QueryObject(parent, 'user', 'edit', 'user',
                                          false, false)
        socket.emit("generateAlternatives", queryObject, true)
        var val = cell.val()
        var caretPos = cell.caret()
        console.log('keyup: ind', ind, 'val', val, 'caretPos', caretPos)
        console.log('val[caretPos-1]', val[caretPos-1])

        // typing in space, that means end of chord symbol, playback chord
        if (val[caretPos-1] == ' ' && e.which!=37 && e.which!=8 && e.which!=46 && e.which!=16) {
//        if (e.which!=37 && e.which!=8 && e.which!=46 && e.which!=16) {
            console.log('...will fire playSubseq, machine', !this.focusMovedByMachine)
            this.editTextNotIndexed = false
            if (!this.focusMovedByMachine) {
                userFocusedCell = true
            }

//            this.emitTextChange(parent)
//            var queryObject = new QueryObject(parent, 'user', 'playSubseq', 'machine',
//                                              true, false)
            var queryObject = new QueryObject(parent, 'user', 'edit', 'user',
                                              true, false)
            var chordSeqsAndFormat = parent.makeChordSeqsAndFormat()
            queryObject.chordSeqsAndFormat = chordSeqsAndFormat
            // for playback??
            console.log("key up fire textChange", 'chordSeqsAndFormat', chordSeqsAndFormat)
            socket.emit("textChange", queryObject)
//            socket.emit("playSubseq", queryObject, false)
        }
    }

//    this.emitTextChange = function(parent) {
//        var queryObject = new QueryObject(parent, 'user', 'edit', 'user',
//                                          true, false)
//        var chordSeqsAndFormat = parent.makeChordSeqsAndFormat()
//        queryObject.chordSeqsAndFormat = chordSeqsAndFormat
//        // for playback??
//        console.log("--- fire textChange", 'chordSeqsAndFormat', chordSeqsAndFormat)
//        this.socket.emit("textChange", queryObject)
////        socket.emit("playSubseq", queryObject, false)
//    }

    this.makeChordSeqsAndFormat = function() {
        var activeIdx = this.getFocusCellIdx()
        var chordSeqsAndFormat = []
        var seq = this.getTextList()
        for (var i=0; i<seq.length; i++) {
            if (activeIdx == i) {
                var format = [seq[i], true]
            } else {
                var format = [seq[i], false]
            }
            chordSeqsAndFormat.push(format)
        }
        return chordSeqsAndFormat
    }




//    this.bind = function() {
//        // display caret pos
//        // need to loop through all of them
//        var cells = this.getCells()
//        cells[0].bind("mouseup",
//            function() {
//                caretPos = $(this).caret().start
//                posInt.text(caretPos)
//        });
//    }
} // function
