---
title: Remote Debugging your AI Agents Using Chrome DevTool MCP on Cloud IDEs
date: 2025-10-18 12:00:00 +0100
categories: [Gen AI, MCP]
tags: [Gen AI, Frontend, Debugging, Chrome DevTools, Cloud IDEs]
image:
  path: /assets/img/chrome-debug.png
  alt: "Remote Debugging your AI Agents Using Chrome DevTool MCP on Cloud IDEs"
---

Cloud IDEs and sandboxes often can’t launch a full Chrome. But the **Chrome DevTools MCP server** lets LLM tools do real browser actions (navigate, trace performance, inspect network/console, run JS) *if* it can attach to a Chrome instance. The trick: attach to a Chrome you run **outside** the sandbox via a secure tunnel—then use Gemini CLI to issue natural-language tasks. In this post, I'll be using **Firebase Studio** as the cloud IDE, but the pattern applies to any environment with Gemini CLI and network access.

---

## What we’ll build

```
Gemini CLI (Firebase Studio)
        |
   (MCP: chrome-devtools)
        |  --browserUrl=https://<PUBLIC_BASE>  → GET /json/version
        |  ...then connect to WSS /devtools/browser/<id>
        v
   [Node “shim” proxy on your machine]
        |  - rewrites Host: 127.0.0.1
        |  - rewrites webSocketDebuggerUrl in /json/version to your tunnel host
        v
   Chrome (headless) with CDP 127.0.0.1:9222
```

Why the shim? The MCP server reads `webSocketDebuggerUrl` *as-is* from `/json/version`. Chrome always advertises `ws://127.0.0.1:9222/...`. In a sandbox, dialing that fails. Our shim rewrites the JSON so the MCP server dials **`wss://<your-public-host>/devtools/browser/<id>`** instead — through your tunnel.

---

## Prerequisites

* A Firebase project (for Hosting; any simple site will do).
* **Firebase Studio** workspace with **Gemini CLI**.
* Node.js on **your** machine (to run the shim).
* **Google Chrome** (or Chromium) on your machine.
* A tunneling tool: **ngrok** or Cloudflare tunnel. There are other solutions for exposing local services - pick your favorite.

---

## Step 0 — (Optional) A tiny “buggy” page to analyze

Here’s a minimal page you can deploy as `public/index.html`:

```html
<!doctype html><meta charset="utf-8">
<title>MCP Debug Demo</title>
<input id="q" placeholder="Type (e.g. chicken)…">
<button id="jank">Force Jank</button>
<pre id="log"></pre>
<script>
let reqs=0, leaks=0;
const log = (...a)=>document.getElementById('log').textContent += a.join(' ')+'\n';
async function search(t){ if(!t) return;
  reqs++; log('fetching', reqs);
  const r = await fetch('https://dummyjson.com/recipes/search?q='+encodeURIComponent(t));
  await r.json();
}
function heavy(){ let s=0; for(let i=0;i<3e7;i++) s+=i%3; log('jank done'); }
const q=document.getElementById('q');
q.addEventListener('input', (e)=>{
  // intentionally bad: new scroll listener each input + no debounce
  window.addEventListener('scroll', ()=>{ let s=0; for(let i=0;i<30000;i++) s+=Math.sqrt(i); });
  leaks++; log('leaks', leaks);
  search(e.target.value);
});
document.getElementById('jank').onclick = heavy;
</script>
```

Deploy with `firebase deploy --only hosting` and note your URL `https://<project-id>.web.app`.

---

## Step 1 — Start Chrome with the DevTools protocol **on your machine**

```bash
# macOS (adjust for Windows/Linux as needed)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --headless=new \
  --user-data-dir=/tmp/mcp-chrome
```

Sanity check:

```bash
curl http://127.0.0.1:9222/json/version  # should return JSON w/ webSocketDebuggerUrl
```

---

## Step 2 — Expose it with a tunnel

> We need two things:
>
> 1. Chrome must see an allowed **Host** header (IP or `localhost`).
> 2. The DevTools **WebSocket** must pass through.

Start ngrok **with Host header rewrite**:

```bash
ngrok http 9222 --host-header="127.0.0.1:9222"
```

Verify from anywhere (including Firebase Studio terminal):

```bash
curl https://<YOUR_NGROK_HOST>/json/version
# Expect JSON (Host-header error = wrong flags)
```

---

## Step 3 — Add the tiny **shim** that rewrites `/json/version`

Create a folder on your machine:

```bash
mkdir -p ~/cdp-shim && cd ~/cdp-shim
npm init -y
npm i http-proxy express
```

Create `server.js`:

```js
// Minimal CDP shim: rewrites webSocketDebuggerUrl to your public host
const express = require('express');
const httpProxy = require('http-proxy');
const http = require('http');

const UPSTREAM = 'http://127.0.0.1:9222';
const PUBLIC_HOST = process.env.PUBLIC_HOST; // e.g. my-tunnel.ngrok-free.app
if (!PUBLIC_HOST) { console.error('Set PUBLIC_HOST to your tunnel host'); process.exit(1); }

const app = express();
const proxy = httpProxy.createProxyServer({ target: UPSTREAM, ws: true, changeOrigin: false });

// 1) Rewrite /json/version to swap scheme+host in webSocketDebuggerUrl
app.get('/json/version', (req, res) => {
  const reqUp = http.request({ host: '127.0.0.1', port: 9222, path: '/json/version',
    headers: { Host: '127.0.0.1:9222' } }, (up) => {
    let body = ''; up.setEncoding('utf8');
    up.on('data', c => body += c);
    up.on('end', () => {
      try {
        const j = JSON.parse(body);
        const path = new URL(j.webSocketDebuggerUrl).pathname; // /devtools/browser/<id>
        j.webSocketDebuggerUrl = `wss://${PUBLIC_HOST}${path}`;
        res.set('content-type', 'application/json; charset=utf-8');
        res.end(JSON.stringify(j));
      } catch (e) { res.statusCode = 502; res.end(String(e)); }
    });
  });
  reqUp.on('error', e => { res.statusCode = 502; res.end(String(e)); });
  reqUp.end();
});

// 2) Everything else → proxy straight through (incl. WS upgrades)
app.use((req, res) => {
  // Ensure Chrome accepts upstream Host:
  req.headers.host = '127.0.0.1:9222';
  proxy.web(req, res, { target: UPSTREAM });
});

const server = app.listen(3000, '127.0.0.1', () =>
  console.log('CDP shim on http://127.0.0.1:3000 (PUBLIC_HOST=%s)', PUBLIC_HOST)
);

server.on('upgrade', (req, socket, head) => {
  req.headers.host = '127.0.0.1:9222';
  proxy.ws(req, socket, head, { target: UPSTREAM });
});
```

Run it:

```bash
PUBLIC_HOST=<YOUR_NGROK_HOST> node server.js
```

Update ngrok to tunnel **the shim** (port 3000), not 9222:

```bash
ngrok http 3000
# Your public base is now https://<YOUR_NGROK_HOST>
curl https://<YOUR_NGROK_HOST>/json/version    # JSON, with wss://<YOUR_NGROK_HOST>/devtools/browser/<id>
```

WebSocket sanity:

```bash
# Optional: require('ws') or use wscat
npx wscat -c wss://<YOUR_NGROK_HOST>/devtools/browser/<id>   # should connect
```

---

## Step 4 — Point **Gemini CLI** (Firebase Studio) at the tunnel

Create or edit **`.gemini/settings.json`** in your workspace:

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "command": "npx",
      "args": [
        "chrome-devtools-mcp@latest",
        "--browserUrl=https://<YOUR_NGROK_HOST>",
        "--headless=true",
        "--isolated=true",
        "--logFile=.gemini/chrome-mcp.log"
      ]
    }
  }
}
```

Restart the Gemini shell and run `/mcp`. You should see `chrome-devtools` with a healthy tool list (navigate_page, list_pages, evaluate_script, list_network_requests, performance_* …).

> Note: `/mcp` only lists tools. The first **real** action (like `new_page` or `navigate_page`) will trigger the `/json/version` fetch and then the **WSS** attach. Watch your ngrok logs to confirm both.

---

## Step 6 — Runtime analysis on your Firebase app

**Find network spam**

> Open `https://<project-id>.web.app`. Emulate Slow 3G. Focus `#q`, type `chicken`, wait 2s, then `list_network_requests`. Count calls to `dummyjson.com/recipes/search` and explain duplication.

**Prove event-listener leak**

> Type `soup` in `#q`, then `list_console_messages` filtering “added scroll listener”, and `evaluate_script` returning `window.__scrollListenerCount`. Summarize what’s wrong.

**Capture jank & get insights**

> `performance_start_trace` → click the “Force Jank” button → after 3s `performance_stop_trace` → `performance_analyze_insight`. Report long tasks & suggestions.

---

## Troubleshooting (copy-paste fixes)

* **“Disconnected (0 tools cached)”**
  Ensure Node ≥ 22 in your Studio environment; restart Gemini after editing `.gemini/settings.json`.

* **QUIC handshake timeouts with Cloudflare**
  Force HTTP/2 (TCP) instead of QUIC (UDP):
  `cloudflared tunnel --protocol http2 --url http://127.0.0.1:9222`

* **“Host header is specified and is not an IP address or localhost.”**
  Your tunnel must send **Host: 127.0.0.1** upstream.

  * ngrok: `--host-header="127.0.0.1:9222"`
  * Cloudflare named tunnel: `originRequest.httpHostHeader: 127.0.0.1`

* **307 on WebSocket handshake**
  Something is redirecting (Access login, path normalization). Disable Access for this host, or use ngrok; ensure you dial **`wss://<host>/devtools/browser/<id>`** exactly.

* **You only see GET `/json/version`, no WSS**
  That’s precisely why we added the shim: MCP read `ws://127.0.0.1:9222/...` and tried to dial it locally. Use the shim and point `--browserUrl` at **the shim’s** public base.

---

## Appendix: Windows commands

**Chrome**

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9222 --headless=new --user-data-dir=$env:TEMP\mcp-chrome
```

**Shim**

```powershell
mkdir cdp-shim; cd cdp-shim
npm init -y
npm i http-proxy express
setx PUBLIC_HOST "<YOUR_NGROK_HOST>"
node server.js
```

**ngrok**

```powershell
ngrok http 3000
```

---

That’s it! You now have a repeatable pattern to drive **real, runtime Chrome debugging** from a cloud workspace using **Gemini CLI + Chrome DevTools MCP**, even when the workspace can’t run Chrome itself.
