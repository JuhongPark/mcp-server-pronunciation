# MCP Directory Submission Checklist

This project is already published to the official MCP Registry under:

```text
io.github.JuhongPark/pronunciation
```

Current public package:

```text
uvx mcp-server-pronunciation@0.3.0
```

## Submission Copy

Use this copy for public MCP directories that require a manual form.

Server name:

```text
Pronunciation & Voice Coach
```

Short description:

```text
Local MCP voice coach with English pronunciation, grammar, fluency, phoneme-level feedback, practice drills, and learner-profile hints.
```

Long description:

```text
Pronunciation & Voice Coach is a local-first MCP server for voice conversation and English learning feedback. It records from the user's microphone, transcribes speech locally with faster-whisper, and returns pronunciation, grammar, fluency, phoneme-level drill feedback, practice suggestions, and learner-profile hints. It supports Claude Desktop, Claude Code, Cursor, VS Code MCP clients, macOS, Linux, Windows, and WSL2.
```

Repository:

```text
https://github.com/JuhongPark/mcp-server-pronunciation
```

Package:

```text
https://pypi.org/project/mcp-server-pronunciation/0.3.0/
```

Install command:

```bash
uvx mcp-server-pronunciation@0.3.0
```

Suggested categories:

```text
Education
Language Learning
Audio Processing
Speech Processing
Local MCP Server
```

Suggested tags:

```text
mcp, model-context-protocol, pronunciation, voice, speech, english-learning, language-learning, whisper
```

## Directory Targets

### Official MCP Registry

Status: published.

The server is active in the official registry. Publish `v0.3.0` to make the
stable package the latest registry version.

### Glama

Status: submit or wait for indexing.

Glama indexes open-source MCP servers and provides server pages with tool
schemas, install instructions, inspection, quality checks, and search. If the
official registry sync does not pick up the project, submit the GitHub
repository URL manually:

```text
https://github.com/JuhongPark/mcp-server-pronunciation
```

### PulseMCP

Status: monitor.

PulseMCP lists MCP servers and shows official registry metadata when available.
If it does not appear after the registry sync window, submit or contact the
site with the repository URL and official registry name.

### Smithery

Status: future work.

Smithery supports URL-based Streamable HTTP servers and local stdio servers
published through MCPB packaging. This project currently ships as a local stdio
Python package, so Smithery publication should wait until one of these is ready:

- an MCPB bundle for the local stdio server.
- a Streamable HTTP transport or hosted endpoint.

### mcpservers.org

Status: manual form.

Use the submission copy above. The form asks for a contact email, which should
be provided by the maintainer rather than committed to this repository.

### Awesome MCP Servers Lists

Status: prepare a pull request for `punkpeye/awesome-mcp-servers`.

Suggested branch:

```text
https://github.com/JuhongPark/awesome-mcp-servers/tree/add-pronunciation-voice-coach
```

Open a pull request against the active upstream:

```text
https://github.com/punkpeye/awesome-mcp-servers/compare/main...JuhongPark:awesome-mcp-servers:add-pronunciation-voice-coach?expand=1
```
