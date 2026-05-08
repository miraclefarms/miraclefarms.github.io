# X Publishing

MiracleFarms can publish a post summary or thread to X after the GitHub.io post is published.

## Setup

1. Create an X Developer project and app.
2. Generate an OAuth 2.0 user access token for the MiracleFarms account with `tweet.write`.
3. Add these values to `.env`:

```bash
X_USER_ACCESS_TOKEN=
X_CLIENT_ID=
X_CLIENT_SECRET=
X_REDIRECT_URI=http://127.0.0.1:8787/callback
X_USERNAME=miraclefarms_io
X_API_BASE_URL=https://api.x.com
X_SITE_URL=https://miraclefarms.github.io
X_DRY_RUN=1
X_POST_MODE=thread
X_AI_TIMEOUT_MS=90000
X_ALLOW_DETERMINISTIC_FALLBACK=0
ENABLE_X_PUSH=0
```

Keep `ENABLE_X_PUSH=0` for the daily pipeline until X content is ready. Set `ENABLE_X_PUSH=1` to enable the optional Stage 8, and keep `X_DRY_RUN=1` until a preview looks right. Set `X_DRY_RUN=0` only when you are ready to post for real.

## OAuth Setup

After setting the app callback URL to `http://127.0.0.1:8787/callback`, run:

```bash
npm run auth:x -- <client-id>
```

The script opens X's OAuth authorization screen, receives the local callback, exchanges the authorization code for a user access token, and writes `X_USER_ACCESS_TOKEN` to `.env`. If the app is configured as a Web App / Automated App, add `X_CLIENT_SECRET` to `.env` before running the script.

## Manual Preview

```bash
npm run publish:x -- _posts/2026-05-08-ai-infra-daily-brief.md --dry-run
```

This command uses the configured `AI_CLI` and `.agents/skills/x-formatter/SKILL.md` to rewrite the GitHub.io article into English-only X copy.

Single-post mode is useful if the X API plan blocks reply-thread creation:

```bash
npm run publish:x -- _posts/2026-05-08-ai-infra-daily-brief.md --dry-run --single
```

## Daily Pipeline

The daily pipeline runs X publishing as optional Stage 8:

```bash
ENABLE_X_PUSH=1 X_DRY_RUN=1 ./ai-morning-report/bin/run-daily.sh
```

When `ENABLE_X_PUSH` is not `1`, the daily pipeline does not call Stage 8 and does not run X rewriting. When enabled, the stage asks the configured AI CLI to rewrite the GitHub.io post through `.agents/skills/x-formatter/SKILL.md`, producing an English-only JSON thread. By default, AI generation failure stops the X stage so Chinese fallback content is not accidentally posted. Set `X_ALLOW_DETERMINISTIC_FALLBACK=1` only for local debugging. Successful real posts are recorded in `.git/x-publish-record.local.json` so the same post is not published twice unless `X_FORCE_POST=1`.

## Notes

X's current Manage Posts API uses `POST /2/tweets` with a user access token. Thread mode posts the first item, then posts each later item as a reply to the previous post. If your API plan rejects replies, switch to `X_POST_MODE=single`.
