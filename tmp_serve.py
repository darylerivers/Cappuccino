import http.server,socketserver,ssl
class ServeHandler(socketserver.WSGIServer):
 def serve_forever(self):
  super().serve_forever()
