import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial

PORT = int(os.environ.get("PORT", "8080"))
handler = partial(SimpleHTTPRequestHandler, directory="docs")
server = ThreadingHTTPServer(("0.0.0.0", PORT), handler)
print(f"Serving research paper from docs/ on port {PORT}")
server.serve_forever()
