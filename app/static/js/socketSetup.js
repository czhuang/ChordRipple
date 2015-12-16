
var socket = undefined;


$(document).ready(function() {

// connecting
  status('Connecting...');
  WEB_SOCKET_DEBUG = true;
  socket = io.connect("");

  socket.on('connect', function() {
    status('Connected.');
    socket.emit('ping', "Hello World!", videoScore.syncEvents.length);
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
});