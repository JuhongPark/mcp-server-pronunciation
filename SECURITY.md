# Security Policy

## Supported Versions

Security fixes target the latest released version and the current `main` branch.
Older releases may receive fixes when the patch is small and the risk is high.

## Reporting A Vulnerability

Please report security issues privately instead of opening a public GitHub issue.

Use GitHub's private vulnerability reporting feature if it is available on this
repository. If it is not available, contact the maintainer through the email
address listed on the maintainer's GitHub profile.

Please include:

- A short description of the issue.
- Steps to reproduce.
- The affected version or commit.
- Whether the issue can expose local audio, transcripts, files, credentials, or
  model cache contents.

## Security Scope

This project records microphone audio locally, writes temporary WAV files, and
loads speech models from local caches. Security-sensitive reports include:

- Unexpected network transmission of audio or transcripts.
- Unsafe handling of temporary audio files.
- Path traversal or unintended file access through tool arguments.
- Dependency behavior that silently changes the privacy model.
- Commands or docs that could expose secrets in public MCP client configs.

## Privacy Expectations

Audio processing is intended to run locally. First-time model and resource
downloads may contact external hosts such as Hugging Face, PyTorch, or NLTK
mirrors. These downloads should be documented and diagnosable.

Do not include private audio samples, transcripts, API keys, or MCP client
configuration secrets in public issues.

