
function ComponentState(id, onPlay, seekTime, timePrecisionSlack) {
    this.id = id
    this.onPlay = onPlay
    this.seekTime = seekTime
    // could be a static that's shared among different instantiations
    this.timePrecisionSlack = timePrecisionSlack
    this.resetThreshold = function() {
        if (this.seekTime > this.timePrecisionSlack) {
            return this.seekTime + this.timePrecisionSlack
        } else {
            // if for example want to reset to 0 and the "callbacks" resolution is not as high
            // might miss 0, so here we're setting this so later on whenever less than this value
            // we'll say we've successfully reset to 0
            return this.timePrecisionSlack
        }
    }
}


function Components(components, lastSeekedComponent) {
    this.components = components
    this.lastSeekedComponent = lastSeekedComponent
    this.onPlay = function(componentId) {
//        console.log("components.onPlay id:", componentId)
        if (componentId == undefined) {
            var state = true
            for (var i=0; i<this.components.length; i++) {
                if (!this.components[i].onPlay) {
                    state = false
                    break
                }
            }
            return state
        } else {
            var state = this.getComponent(componentId).onPlay
//            console.log('all onPlay state', state)
            return state
        }
    } // onPlay(componentId) {

    this.setToPlay = function(componentId) {
//        console.log('components.setToPlay:', componentId)
        if (componentId == undefined) {
            for (var i=0; i<this.components.length; i++) {
                this.components[i].onPlay = true
            }
        } else {
            comp = this.getComponent(componentId)
            comp.onPlay = true
        }
    }

    this.getComponent = function(id) {
        // can not use id to index the list because they are not contiguous
        for (var i=0; i<this.components.length; i++) {
            if (this.components[i].id == id) {
                return this.components[i]
            }
        }
        return undefined
    }

    this.setToStop = function(componentId) {
//        console.log('components.setToStop:', componentId)
        if (componentId == undefined) {
            for (var i=0; i<this.components.length; i++) {
                this.components[i].onPlay = false
            }
        } else {
            comp = this.getComponent(componentId)
            comp.onPlay = false
        }
    }

    this.onPlayAndSeeked = function(componentId) {
        comp = this.getComponent(componentId)
        return (comp.onPlay && lastSeekedComponent == componentId)
    }
}

