#!/usr/bin/env python3
"""
Simple HTTP server for The-BoundedGlitchGPT Bot
Provides REST API for model interaction.
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys
from pathlib import Path

from bot import BoundedGlitchBot


class BotHandler(BaseHTTPRequestHandler):
    """HTTP request handler for bot API."""
    
    # Class variable to hold bot instance
    bot = None
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        
        if parsed_url.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>The-BoundedGlitchGPT</title>
                <style>
                    body { font-family: monospace; max-width: 600px; margin: 50px auto; }
                    #chat { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: auto; margin-bottom: 10px; }
                    .message { margin: 5px 0; }
                    .user { color: blue; }
                    .bot { color: green; }
                    input { width: 80%; padding: 5px; }
                    button { padding: 5px 10px; }
                </style>
            </head>
            <body>
                <h1>The-BoundedGlitchGPT</h1>
                <div id="chat"></div>
                <input type="text" id="prompt" placeholder="Enter prompt...">
                <button onclick="sendMessage()">Send</button>
                
                <script>
                    function sendMessage() {
                        const prompt = document.getElementById('prompt').value;
                        if (!prompt) return;
                        
                        const chatDiv = document.getElementById('chat');
                        chatDiv.innerHTML += '<div class="message user"><b>You:</b> ' + prompt + '</div>';
                        document.getElementById('prompt').value = '';
                        
                        fetch('/api/chat?prompt=' + encodeURIComponent(prompt))
                            .then(r => r.json())
                            .then(data => {
                                chatDiv.innerHTML += '<div class="message bot"><b>Bot:</b> ' + data.response + '</div>';
                                chatDiv.scrollTop = chatDiv.scrollHeight;
                            });
                    }
                    
                    document.getElementById('prompt').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') sendMessage();
                    });
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        
        elif parsed_url.path == '/api/chat':
            params = parse_qs(parsed_url.query)
            prompt = params.get('prompt', [''])[0]
            
            if not prompt:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No prompt provided'}).encode())
                return
            
            response = self.bot.chat(prompt, max_length=100, temperature=0.8)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'response': response}).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress logging."""
        pass


def main():
    """Start the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="The-BoundedGlitchGPT Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--data-dir", default="data", help="Directory containing training data")
    
    args = parser.parse_args()
    
    # Initialize bot
    print("[*] Initializing bot...")
    bot = BoundedGlitchBot(data_dir=args.data_dir)
    bot.startup()
    
    # Set class variable
    BotHandler.bot = bot
    
    # Start server
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, BotHandler)
    
    print(f"\n[✓] Server running at http://{args.host}:{args.port}")
    print("[*] Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped")


if __name__ == "__main__":
    main()
