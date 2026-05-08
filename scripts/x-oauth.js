#!/usr/bin/env node

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const dotenv = require('dotenv');

const PROJECT_ROOT = path.join(__dirname, '..');
const ENV_PATH = path.join(PROJECT_ROOT, '.env');
dotenv.config({ path: ENV_PATH });

const DEFAULT_REDIRECT_URI = 'http://127.0.0.1:8787/callback';
const DEFAULT_SCOPES = ['tweet.read', 'tweet.write', 'users.read', 'offline.access'];

function base64Url(buffer) {
  return buffer
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

function makeVerifier() {
  return base64Url(crypto.randomBytes(48));
}

function makeChallenge(verifier) {
  return base64Url(crypto.createHash('sha256').update(verifier).digest());
}

function upsertEnvVars(filePath, updates) {
  const existing = fs.existsSync(filePath) ? fs.readFileSync(filePath, 'utf8') : '';
  const lines = existing.split(/\r?\n/);
  const seen = new Set();
  const nextLines = lines.map((line) => {
    const match = line.match(/^([A-Z0-9_]+)=/);
    if (!match || !Object.prototype.hasOwnProperty.call(updates, match[1])) {
      return line;
    }
    seen.add(match[1]);
    return `${match[1]}=${updates[match[1]]}`;
  });

  for (const [key, value] of Object.entries(updates)) {
    if (!seen.has(key)) {
      nextLines.push(`${key}=${value}`);
    }
  }

  fs.writeFileSync(filePath, `${nextLines.filter((line, index, arr) => index < arr.length - 1 || line).join('\n')}\n`, 'utf8');
}

function openUrl(url) {
  try {
    execFileSync('open', [url], { stdio: 'ignore' });
  } catch {
    console.log(`Open this URL manually:\n${url}`);
  }
}

function waitForCallback(port, expectedState) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url, `http://127.0.0.1:${port}`);
      if (url.pathname !== '/callback') {
        res.writeHead(404);
        res.end('Not found');
        return;
      }

      const error = url.searchParams.get('error');
      if (error) {
        res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end(`X authorization failed: ${error}`);
        server.close();
        reject(new Error(`X authorization failed: ${error}`));
        return;
      }

      const state = url.searchParams.get('state');
      if (state !== expectedState) {
        res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Invalid OAuth state.');
        server.close();
        reject(new Error('Invalid OAuth state'));
        return;
      }

      const code = url.searchParams.get('code');
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end('<h1>MiracleFarms X authorization complete</h1><p>You can close this tab and return to Codex.</p>');
      server.close();
      resolve(code);
    });

    server.on('error', reject);
    server.listen(port, '127.0.0.1', () => {
      console.log(`[x-oauth] Listening on http://127.0.0.1:${port}/callback`);
    });
  });
}

async function exchangeCode({ clientId, clientSecret, code, verifier, redirectUri }) {
  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    redirect_uri: redirectUri,
    code_verifier: verifier,
    code,
  });
  const headers = { 'Content-Type': 'application/x-www-form-urlencoded' };

  if (clientSecret) {
    headers.Authorization = `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`;
  } else {
    body.set('client_id', clientId);
  }

  const response = await fetch('https://api.x.com/2/oauth2/token', {
    method: 'POST',
    headers,
    body,
  });

  const json = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`Token exchange failed: ${response.status} ${JSON.stringify(json).slice(0, 500)}`);
  }
  if (!json.access_token) {
    throw new Error(`Token exchange did not return access_token: ${JSON.stringify(json).slice(0, 500)}`);
  }
  return json;
}

async function main() {
  const clientId = process.argv[2] || process.env.X_CLIENT_ID;
  if (!clientId) {
    console.error('Usage: node scripts/x-oauth.js <X_CLIENT_ID>');
    process.exit(1);
  }

  const redirectUri = process.env.X_REDIRECT_URI || DEFAULT_REDIRECT_URI;
  const clientSecret = process.env.X_CLIENT_SECRET || '';
  const port = Number.parseInt(new URL(redirectUri).port || '80', 10);
  const verifier = makeVerifier();
  const challenge = makeChallenge(verifier);
  const state = base64Url(crypto.randomBytes(24));
  const authUrl = new URL('https://x.com/i/oauth2/authorize');
  authUrl.searchParams.set('response_type', 'code');
  authUrl.searchParams.set('client_id', clientId);
  authUrl.searchParams.set('redirect_uri', redirectUri);
  authUrl.searchParams.set('scope', DEFAULT_SCOPES.join(' '));
  authUrl.searchParams.set('state', state);
  authUrl.searchParams.set('code_challenge', challenge);
  authUrl.searchParams.set('code_challenge_method', 'S256');

  const callbackPromise = waitForCallback(port, state);
  console.log('[x-oauth] Opening X authorization page...');
  openUrl(authUrl.toString());
  const code = await callbackPromise;

  console.log('[x-oauth] Exchanging authorization code...');
  const token = await exchangeCode({ clientId, clientSecret, code, verifier, redirectUri });

  upsertEnvVars(ENV_PATH, {
    X_CLIENT_ID: clientId,
    X_CLIENT_SECRET: clientSecret,
    X_REDIRECT_URI: redirectUri,
    X_USER_ACCESS_TOKEN: token.access_token,
    X_REFRESH_TOKEN: token.refresh_token || process.env.X_REFRESH_TOKEN || '',
    X_DRY_RUN: process.env.X_DRY_RUN || '1',
    ENABLE_X_PUSH: process.env.ENABLE_X_PUSH || '0',
  });

  console.log(`[x-oauth] Token saved to ${ENV_PATH}`);
  console.log('[x-oauth] Keep X_DRY_RUN=1 until preview looks good.');
  process.exit(0);
}

main().catch((err) => {
  console.error('[x-oauth] Failed:', err.message);
  process.exit(1);
});
