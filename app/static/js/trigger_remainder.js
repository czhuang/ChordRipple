

    // the list to store all the notes to be played
    function notesToPlay(currentVideoTime, justSeeked) {
    // strip out the notes that are too late from the notes
        if (currentVideoTime == undefined) {
            return [[], currentVideoTime]
        }
        console.log('---notesToPlay', currentVideoTime)
        console.log('onset times', getOnsetList(notesRemainingInLoop))
        var notesTooLate = []
        var notesForLater = []
        var originalNotesRemainingInLoop = notesRemainingInLoop.length
        var i = 0;
        while (notesRemainingInLoop.length > 0 && i < originalNotesRemainingInLoop) {
            if (notesRemainingInLoop[notesRemainingInLoop.length-1].onset <= currentVideoTime) {
                notesTooLate.push(notesRemainingInLoop.pop())
            } else{
                notesForLater.push(notesRemainingInLoop.pop())
            }
            i += 1
        }
        notesRemainingInLoop = notesForLater
        console.log('# of notesTooLate', notesTooLate.length)
        console.log('# of notes', notesRemainingInLoop.length)

        var currentMidiTime = currentVideoTime
        // if there are notes that are too late that means
        // the actual midi note time is behind
        if (notesTooLate.length > 0) {
            currentMidiTime = getMinOnsetTime(notesTooLate)
        }
        console.log('...notesToPlay, updated currentVideoTime', currentVideoTime)
        if (justSeeked) {
            return [[], currentMidiTime]
        } else {
            return [notesTooLate, currentMidiTime]
        }
    }
