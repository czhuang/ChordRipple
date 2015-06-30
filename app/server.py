from gevent import monkey
monkey.patch_socket()
from flask import Flask, request, send_file
from socketio import socketio_manage
from resource import Resource


# Flask routes
app = Flask(__name__)


@app.route('/')
def index():
    return send_file('static/index.html')


@app.route("/socket.io/<path:path>")
def run_socketio(path):
    socketio_manage(request.environ, {'': Resource})


if __name__ == '__main__':
    port_num = 8088
    print 'Listening on http://localhost:%d' % port_num
    app.debug = True
    import os
    from werkzeug.wsgi import SharedDataMiddleware
    app = SharedDataMiddleware(app,
                               {'/': os.path.join(os.path.dirname(__file__),
                                                  'static')})
    from socketio.server import SocketIOServer
    SocketIOServer(('0.0.0.0', port_num), app,
                   resource="socket.io", policy_server=False).serve_forever()
