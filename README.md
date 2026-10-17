# ai-lab: Local LLM Server on Bazzite

> Work in progress experiment, your here for the experiment and learnings.

Runs a local LLM server on Bazzite (immutable Fedora) using podman and CDI
(Container Device Interface) for NVIDIA GPU passthrough. The server is
llama.cpp (`ghcr.io/ggml-org/llama.cpp:server-cuda`) serving an
OpenAI-compatible API at `http://localhost:58080`.

## Requirements

- Bazzite (Silverblue) or any SELinux-enabled Fedora variant with an NVIDIA GPU
- NVIDIA vendor driver installed on the host
- nvidia-container-toolkit, with a CDI spec present at /etc/cdi/nvidia.yaml
- podman 5.x or newer
- podman-compose
- Enough free disk for the model files you intend to use (tens of GB; see
  models/README.md)

Bazzite ships the NVIDIA driver and nvidia-container-toolkit preinstalled,
which automatically generates the CDI spec. No extra runtime configuration
is needed.

### How GPU passthrough works here

Instead of Docker's `nvidia` runtime or `--gpus all`, podman maps GPU devices
through the CDI spec at /etc/cdi/nvidia.yaml. The compose file uses:

```yaml
devices:
  - nvidia.com/gpu=all
```

Check what device names are available:

```bash
grep -E '^\s*- name:' /etc/cdi/nvidia.yaml
```

`nvidia.com/gpu=all` passes every GPU, `nvidia.com/gpu=0` passes the first
one, and you can pin a specific card by its UUID (shown in the same list).

## Setup

### 1. Verify prerequisites

```bash
nvidia-smi                          # GPU + driver visible on the host
test -f /etc/cdi/nvidia.yaml && echo "CDI spec OK"
podman --version
podman-compose --version
```

If the CDI spec is missing, generate it:

```bash
sudo nvidia-ctk cdi generate
```

### 2. Get a model

Place GGUF model files under `./models/`. The compose file loads whichever
file is set in `MODEL_FILE` (see Configuration below). Pick a quantization
that fits your GPU's VRAM with headroom for context/KV cache.

### 3. Pull the server image

```bash
podman pull ghcr.io/ggml-org/llama.cpp:server-cuda
```

This is the official CUDA build of llama.cpp. It links against the CUDA
driver exposed through CDI; no NVIDIA toolkit is needed inside the
container itself.

### 4. Start the server

```bash
podman compose up -d
```

### 5. Verify it's running

```bash
curl -s http://localhost:58080/health
curl -s http://localhost:58080/v1/models | jq
```

`health` should return `{"status":"ok"}`, and `v1/models` should list your
configured model alias.

## Configuration

Most settings are environment variables, overridable via a `.env` file in the
project root or inline on the command line. Defined in compose.yaml:

- MODEL_NAME: alias the model is served under (`--alias`)
- MODEL_FILE: GGUF file under ./models to load
- NUM_CTX: total context size across all parallel slots (`--ctx-size`)
- NUM_GPU: layers offloaded to GPU (`--n-gpu-layers`; 999 = all)
- NUM_PARALLEL: number of concurrent request slots (`--parallel`)
- NUM_THREAD: host CPU threads (defaults to all cores if unset)
- SLEEP_IDLE_SECONDS: unload the model from GPU memory after this many
  seconds of inactivity
- API_KEY: required. Clients must send it as `Authorization: Bearer <key>`.
  Set it in a local `.env` file (not committed to version control); there is
  no default
- CORS_ORIGINS: allowed origins for browser-based cross-origin requests
  (default restricts this to the server's own origin)

`NUM_CTX` is a pool shared across `NUM_PARALLEL` slots, not a per-conversation
limit. Each slot gets roughly `NUM_CTX / NUM_PARALLEL` tokens of context. The
default here is `NUM_PARALLEL=1`, so a single conversation gets the full
context window; concurrent users queue behind each other rather than each
getting a smaller slice. Raise it temporarily if you want multiple people to
use the server at once, at the cost of less context per conversation, and
both values are bounded by your GPU's VRAM. If the model fails to load with
an out-of-memory error after changing either value, lower one of them.

If you use this server with an agentic client (such as goose), the client's
own model configuration needs a context length matching the effective
per-slot context (`NUM_CTX / NUM_PARALLEL`), a model name matching
`MODEL_NAME` exactly, and the API key configured for authentication. Keep all
of this in sync whenever you change the server's settings.

A few protections are fixed directly in compose.yaml rather than exposed as
environment variables: a CPU core limit, a memory limit, a process-count
limit, and disabling the slot-introspection endpoint. These exist to contain
the blast radius of a misbehaving or malicious client on a shared local
network, adjust them in compose.yaml directly if your hardware or threat
model differs.

## Using the API

The server exposes an OpenAI-compatible API, authenticated with the API key
from your `.env` file:

- Base URL: http://localhost:58080
- POST /v1/chat/completions
- GET /v1/models

```bash
curl -s http://localhost:58080/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -d '{
    "model": "qwen3.8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }' | jq .choices[0].message.content
```

Point any OpenAI-compatible client at this base URL, with the API key set as
a bearer token. Only the `/v1/*` routes are implemented; there is no Ollama
wire-protocol support (`/api/chat`, `/api/generate`) on this image.

### Reaching it from other devices on your network

The server listens on all interfaces, so any device on your local network can
reach it at `http://<server-hostname-or-IP>:58080`, using the same API key.

#### Using mDNS for easier access

With mDNS (multicast DNS) enabled on the server, devices on your local
network can reach it at `http://ai-lab.local:58080` without needing to know
its IP address or configure anything on your router. Most operating systems
(macOS, Linux, and recent Windows) support mDNS out of the box.

**Setting up mDNS on Bazzite:**

Avahi is typically preinstalled and enabled on Bazzite. Verify with:

```bash
systemctl status avahi-daemon
```

If not running, enable it with:

```bash
sudo systemctl enable --now avahi-daemon
```

If a device on your network doesn't support mDNS, use the server's IP address
instead, or consult your router or network controller's documentation for
setting up a local DNS entry or a reserved IP address.

## Using goose with this setup

[goose](https://github.com/aaif-goose/goose) is an open-source agentic
CLI/Desktop assistant that can use this server as a model provider.

### Install

```bash
curl -fsSL https://github.com/aaif-goose/goose/releases/download/stable/download_cli.sh | bash
```

To skip interactive setup during install:

```bash
curl -fsSL https://github.com/aaif-goose/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash
```

Keep it up to date:

```bash
goose update
```

### Add this server as a custom provider

Create a JSON file at `~/.config/goose/custom_providers/custom_localhost.json`:

```json
{
  "name": "custom_localhost",
  "engine": "openai",
  "display_name": "Localhost",
  "description": "Custom Localhost provider",
  "base_url": "http://ai-lab.local:58080/v1/chat/completions",
  "requires_auth": true,
  "api_key_env": "API_KEY",
  "supports_streaming": true,
  "models": [
    {
      "name": "qwen3.8",
      "context_limit": 65536
    }
  ]
}
```

Use `engine: "openai"`, since this server only implements the
OpenAI-compatible API. Set `models[].name` to exactly match `MODEL_NAME`
from compose.yaml, and `context_limit` to the effective per-slot context
(`NUM_CTX / NUM_PARALLEL`), not the raw `NUM_CTX` value. Export the API key
as the environment variable named in `api_key_env` before starting goose:

```bash
export API_KEY=<the value from .env>
```

### Select the provider/model

Interactively:

```bash
goose configure
```

Via environment variables:

```bash
export GOOSE_PROVIDER=custom_localhost
export GOOSE_MODEL=qwen3.8
goose session
```

One-off, without changing your default config:

```bash
goose run --provider custom_localhost --model qwen3.8 -t "Hello from the home lab"
```

In an interactive session, switch providers/models at any time with:

```
/model --provider custom_localhost qwen3.8
```

## Security considerations for a shared network

This server is designed to be reachable by every device on your local
network, including ones you don't fully control, rather than isolated on a
dedicated segment. That tradeoff is deliberate for convenience, but it means
the application itself, not network topology, is the main access control.
The following are already in place:

- An API key is required for every request. Without it, requests are
  rejected before they reach the model. Treat this key like a password:
  keep it out of version control, and rotate it if you believe it has
  leaked.
- Cross-origin browser requests are restricted, so a malicious or
  compromised web page loaded on another device on your network cannot
  silently use your API key or call the server on a visitor's behalf.
- The slot-introspection endpoint is disabled, so other devices on the
  network cannot observe in-flight prompts from other users.
- The container has fixed CPU, memory, and process-count limits, so a
  misbehaving or malicious client cannot exhaust host resources beyond
  those bounds.
- `NUM_PARALLEL` caps how many requests can run at once, which limits how
  much total throughput any combination of clients, legitimate or not, can
  draw from the server at the same time.
- The model is unloaded from GPU memory after a period of inactivity
  (`SLEEP_IDLE_SECONDS`), limiting the window during which compute is
  actively available.

These controls assume the clients that do have the API key are trusted.
Anyone with the key can use the full capacity of the server within the
limits above; the key is what distinguishes a trusted device on your network
from an arbitrary one, not VLAN placement or firewall rules alone.

## SELinux note

The host runs SELinux Enforcing. The compose file sets:

```yaml
security_opt:
  - label=disable
```

This runs the container unconfined, which is simplest for a local, trusted
single-user box. If you prefer to keep SELinux enforcing, remove
`security_opt` and instead relabel the models volume in the compose file:

```yaml
volumes:
  - ./models:/models:ro,z
```

## Stopping / cleanup

```bash
podman compose down              # stop and remove container + network
podman compose down --rmi local  # also remove the image
```

