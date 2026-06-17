#!/usr/bin/env node
/**
 * Run Slidev against one of the child decks in presentations/src/.
 *
 * Each deck is a flat entry file `presentations/src/<deck>.md`. They all share
 * the same theme, addon (`@/shared/slidev-addon-base`), `style.css`, figures,
 * and public assets: the Slidev project root is `src/`, and `src/` contains
 * symlinks (`shared`, `style.css`, `public`) back to the shared resources at
 * the presentations root, so `@/shared/...`, auto-loaded `style.css`, and
 * `/figures/...` URLs all resolve identically for every deck.
 *
 * The deck name is the first positional argument; it defaults to `main`.
 * Extra Slidev flags are forwarded after the deck name.
 *
 * Usage:
 *   npm run dev                     # deck = main
 *   npm run dev main
 *   npm run dev modelrecovery
 *   npm run dev modelrecovery -- --port 3030
 *   npm run build modelcomparison
 *   npm run export gazeanalysis
 *
 * (npm forwards args after `npm run <script>`; a leading `--` separator is
 *  tolerated wherever it lands.)
 */
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const srcDir = path.join(root, 'src');
const slidevEntry = path.join(root, 'node_modules', '@slidev', 'cli', 'bin', 'slidev.mjs');

const DEFAULT_DECK = 'main';

/** List available deck names: every `*.md` directly under src/. */
function availableDecks() {
  if (!fs.existsSync(srcDir)) return [];
  return fs
    .readdirSync(srcDir)
    .filter((f) => f.endsWith('.md'))
    .map((f) => f.slice(0, -'.md'.length))
    .sort();
}

function fail(msg) {
  console.error(msg);
  process.exit(1);
}

function main() {
  const cmd = process.argv[2];
  if (!['dev', 'build', 'export'].includes(cmd)) {
    fail('Usage: npm run <dev|build|export> [deck] [-- slidev args...]');
  }

  // Everything after the subcommand. npm may inject a `--` separator; strip
  // any standalone `--` tokens so `npm run dev modelrecovery` and
  // `npm run dev -- modelrecovery` both work. The deck name is the first
  // remaining token that is not a flag; the rest are forwarded to slidev.
  const rest = process.argv.slice(3).filter((a) => a !== '--');

  const decks = availableDecks();
  if (decks.length === 0) {
    fail(`No decks found in ${path.relative(process.cwd(), srcDir)} (expected src/<deck>.md).`);
  }

  let deck = DEFAULT_DECK;
  let forwarded = rest;
  if (rest.length > 0 && !rest[0].startsWith('-')) {
    deck = rest[0];
    forwarded = rest.slice(1);
  }

  if (!decks.includes(deck)) {
    fail(
      `Unknown deck "${deck}".\n` +
        `Available decks: ${decks.join(', ')}\n` +
        `Run, e.g.: npm run ${cmd} ${decks[0]}`,
    );
  }

  const entry = path.join(srcDir, `${deck}.md`);
  if (!fs.existsSync(entry)) {
    fail(`Missing ${path.relative(process.cwd(), entry)}`);
  }

  if (!fs.existsSync(slidevEntry)) {
    fail('Slidev CLI not found. Run npm install in presentations/.');
  }

  // `dev` opens the browser by default, matching the old per-deck script.
  const devFlags = cmd === 'dev' ? ['--open'] : [];
  const slidevArgs =
    cmd === 'dev'
      ? [entry, ...devFlags, ...forwarded]
      : [cmd, entry, ...forwarded];

  const r = spawnSync(process.execPath, [slidevEntry, ...slidevArgs], {
    stdio: 'inherit',
    cwd: root,
    env: process.env,
  });
  process.exit(r.status ?? 1);
}

main();
