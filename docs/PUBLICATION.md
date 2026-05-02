# Publication Checklist

This repository being public on GitHub is only the source-code publication
step. For users to discover and install the beta, publish in three places:

1. PyPI for the Python package.
2. GitHub Releases for the beta release page and notes.
3. MCP Registry for MCP-server discovery metadata.

## Current Status

As of the public beta preparation:

- GitHub repository: public.
- PyPI project: not published yet.
- GitHub Releases: no release yet.
- MCP Registry entry: not published yet.

The prepared beta version is `0.3.0b1`, and the MCP server name is
`io.github.juhongpark/pronunciation`.

## PyPI Registration

This must be done from the maintainer's PyPI account.

The recommended route is PyPI Trusted Publishing with a pending publisher. It
lets the GitHub Actions workflow create the PyPI project on first publish
without storing an API token in GitHub.

Create a pending Trusted Publisher in PyPI with:

- PyPI project name: `mcp-server-pronunciation`
- Owner: `JuhongPark`
- Repository name: `mcp-server-pronunciation`
- Workflow filename: `release.yml`
- Environment name: `pypi`

Official PyPI docs:

- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/

After that PyPI account setup is complete, pushing tag `v0.3.0b1` will trigger
the release workflow and publish:

```bash
git tag v0.3.0b1
git push origin v0.3.0b1
```

Verify:

```bash
curl -s https://pypi.org/pypi/mcp-server-pronunciation/json
```

## GitHub Release

Create a pre-release for the beta:

```bash
gh release create v0.3.0b1 \
  --title "v0.3.0b1 (public beta)" \
  --notes-file docs/releases/v0.3.0b1.md \
  --prerelease
```

## MCP Registry

The MCP Registry hosts metadata, not the package artifact. Publish to PyPI
first, then publish `server.json` to the MCP Registry.

This repository includes the required PyPI ownership marker in `README.md`:

```text
mcp-name: io.github.juhongpark/pronunciation
```

The release workflow publishes `server.json` to the MCP Registry after the
PyPI publish job succeeds. It uses GitHub OIDC authentication and does not need
a registry token secret.

Verify registry publication with:

```bash
curl -s "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.juhongpark/pronunciation"
```

Official MCP Registry docs:

- https://modelcontextprotocol.io/registry/about
- https://modelcontextprotocol.io/registry/package-types
- https://modelcontextprotocol.io/registry/authentication
- https://modelcontextprotocol.io/registry/github-actions
- https://modelcontextprotocol.io/registry/versioning

## Order Of Operations

1. Confirm checks pass locally.
2. Create the pending Trusted Publisher on PyPI.
3. Push tag `v0.3.0b1`.
4. Wait for GitHub Actions to publish to PyPI and then the MCP Registry.
5. Create or verify the GitHub pre-release.
6. Verify PyPI and MCP Registry URLs.

