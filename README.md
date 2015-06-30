ChordRipple
===========
This web app is a search-box like interface for writing chord progressions.  Just type in chord names, and you'll see its ripples propagate down the list of autocompletions.  See substitutions, what's next, surrounding chords change, and continuations that lead you to the end of a phrase.

On a non-snowy day, you might be able to try it out at:
	
	https://chordripple.iis-dev.seas.harvard.edu


INSTALLATION
============

For Mac:

	pip install numpy
	pip install music21
	pip install flask
	pip install gevent-socketio

RUNNING
=======

	cd ChordRipple/app
	python server.py
	
In local browser:

	http://localhost:8088


Some other libraries used:
* http://twitter.github.com/bootstrap/
* https://github.com/mudcube/MIDI.js/
* https://github.com/kpdecker/jsdiff
* https://code.google.com/p/jcaret/
* https://github.com/HIPS/Kayak
* https://github.com/piskvorky/gensim/

REFERENCES
==========
This app uses annotated data from 

De Clercq, Trevor, and David Temperley. "A corpus analysis of rock harmony." Popular Music 30.01 (2011): 47-70.


The model is based on

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).  Distributed Representations of Words and Phrases and their Compositionality. Advances in Neural Information Processing Systems, 26, 3111–3119.



