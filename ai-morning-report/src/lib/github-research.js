const { execFileSync } = require('node:child_process');

const DEFAULT_LIMITS = {
  pull_requests: 20,
  releases: 5,
  commits: 10,
};

const DEFAULT_PRIORITY_WEIGHTS = {
  default_path: 6,
  runtime: 4,
  performance: 4,
  feature: 2,
  stability: 2,
  docs_only: -6,
  ci_only: -5,
  tests_only: -5,
};

const DEFAULT_SIGNAL_TERMS = {
  default_path: [
    'default path',
    'default-path',
    'hot path',
    'critical path',
  ],
  runtime: [
    'runtime',
    'serving',
    'scheduler',
    'kernel',
    'executor',
    'engine',
    'kv cache',
    'moe',
    'quantization',
    'throughput',
  ],
  performance: [
    'performance',
    'latency',
    'throughput',
    'optimize',
    'optimization',
    'faster',
    'speedup',
    'benchmark',
    'cache',
    'memory',
  ],
  feature: [
    'feature',
    'add ',
    'support',
    'enable',
    'introduce',
    'launch',
  ],
  stability: [
    'fix',
    'stability',
    'correctness',
    'reliability',
    'fallback',
    'crash',
    'panic',
  ],
  docs_only: [
    'docs',
    'documentation',
    'readme',
    'changelog',
  ],
  ci_only: [
    'ci',
    'workflow',
    'github actions',
    'buildkite',
    'pipeline',
  ],
  tests_only: [
    'test',
    'tests',
    'pytest',
    'unit test',
    'integration test',
    'coverage',
    'snapshot',
  ],
};

function normalizeRepoName(repo) {
  if (typeof repo === 'string' && repo) {
    return repo;
  }

  if (repo && typeof repo === 'object') {
    if (repo.name) {
      return repo.name;
    }

    if (repo.owner && repo.repo) {
      return `${repo.owner}/${repo.repo}`;
    }
  }

  throw new Error('repo is required');
}

function asLabelNames(labels) {
  return (labels || []).map((label) => {
    if (typeof label === 'string') {
      return label;
    }

    return label && label.name ? label.name : '';
  }).filter(Boolean);
}

function mergeSignalTerms(priorityConfig = {}) {
  return {
    ...DEFAULT_SIGNAL_TERMS,
    ...(priorityConfig.signal_terms || {}),
  };
}

function mergePriorityWeights(priorityConfig = {}) {
  return {
    ...DEFAULT_PRIORITY_WEIGHTS,
    ...(priorityConfig.weights || {}),
  };
}

function detectSignals(candidate, priorityConfig = {}) {
  const labels = asLabelNames(candidate.labels);
  const text = [
    candidate.title || '',
    candidate.summary || '',
    candidate.body || '',
    labels.join(' '),
  ].join(' ').toLowerCase();

  const signalTerms = mergeSignalTerms(priorityConfig);
  const signals = {};

  for (const [signal, terms] of Object.entries(signalTerms)) {
    signals[signal] = terms.some((term) => text.includes(term.toLowerCase()));
  }

  const positiveSignalNames = ['default_path', 'runtime', 'performance', 'feature', 'stability'];
  const hasPositiveSignal = positiveSignalNames.some((name) => signals[name]);

  if (signals.docs_only && hasPositiveSignal) {
    signals.docs_only = false;
  }

  if (signals.ci_only && hasPositiveSignal) {
    signals.ci_only = false;
  }

  if (signals.tests_only && hasPositiveSignal) {
    signals.tests_only = false;
  }

  return signals;
}

function rankCandidatePriority(candidate, priorityConfig = {}) {
  const weights = mergePriorityWeights(priorityConfig);
  const signals = candidate.signals || detectSignals(candidate, priorityConfig);

  return Object.entries(weights).reduce((score, [signal, weight]) => (
    signals[signal] ? score + weight : score
  ), 0);
}

function assertNonEmptyResearch(repoResults) {
  if (!Array.isArray(repoResults) || repoResults.length === 0) {
    throw new Error('no research candidates collected');
  }

  const candidateCount = repoResults.reduce(
    (count, repoResult) => count + ((repoResult.pull_requests || []).length),
    0,
  );

  if (candidateCount > 0) {
    return repoResults;
  }

  const allFailed = repoResults.every((repoResult) => (
    Array.isArray(repoResult.failures) && repoResult.failures.length > 0
  ));

  if (allFailed) {
    throw new Error('all upstream collection failed; no research candidates collected');
  }

  throw new Error('no research candidates collected');
}

function createGhClient({
  execFile = execFileSync,
  timeoutMs = 30000,
} = {}) {
  function runGh(args) {
    const output = execFile('gh', args, {
      encoding: 'utf8',
      timeout: timeoutMs,
      maxBuffer: 1024 * 1024 * 10,
    });

    return output ? JSON.parse(output) : [];
  }

  return {
    listMergedPullRequests(repo, window, limit = DEFAULT_LIMITS.pull_requests) {
      return runGh([
        'pr',
        'list',
        '--repo',
        repo,
        '--state',
        'merged',
        '--search',
        `merged:${window.startDate}..${window.endDate}`,
        '--limit',
        String(limit),
        '--json',
        'number,title,url,mergedAt,labels,author,body,additions,deletions,changedFiles',
      ]);
    },

    listReleases(repo, limit = DEFAULT_LIMITS.releases) {
      return runGh([
        'api',
        `repos/${repo}/releases?per_page=${limit}`,
      ]);
    },

    listCommits(repo, window, limit = DEFAULT_LIMITS.commits) {
      const since = encodeURIComponent(window.start);
      const until = encodeURIComponent(window.end);
      return runGh([
        'api',
        `repos/${repo}/commits?since=${since}&until=${until}&per_page=${limit}`,
      ]);
    },
  };
}

function normalizePullRequest(pr, repo, priorityConfig = {}) {
  const labels = asLabelNames(pr.labels);
  const signals = detectSignals({
    title: pr.title,
    body: pr.body,
    labels,
  }, priorityConfig);

  return {
    type: 'pull_request',
    repo,
    number: pr.number,
    title: pr.title,
    url: pr.url,
    merged_at: pr.mergedAt || pr.merged_at,
    author: pr.author && pr.author.login ? pr.author.login : null,
    labels,
    body: pr.body || '',
    additions: pr.additions ?? null,
    deletions: pr.deletions ?? null,
    changed_files: pr.changedFiles ?? pr.changed_files ?? null,
    signals,
    priority_score: rankCandidatePriority({ labels, signals }, priorityConfig),
    evidence: [{ kind: 'pull_request', url: pr.url }],
  };
}

function normalizeRelease(release, repo) {
  return {
    type: 'release',
    repo,
    tag_name: release.tag_name,
    title: release.name || release.tag_name,
    url: release.html_url || release.url,
    published_at: release.published_at || release.created_at || null,
  };
}

function normalizeCommit(commit, repo) {
  const commitMessage = commit.commit && commit.commit.message
    ? commit.commit.message.split('\n')[0]
    : commit.message;

  return {
    type: 'commit',
    repo,
    sha: commit.sha,
    title: commitMessage || '',
    url: commit.html_url || commit.url,
    authored_at: commit.commit && commit.commit.author ? commit.commit.author.date : null,
    author: commit.commit && commit.commit.author ? commit.commit.author.name : null,
  };
}

async function collectRepoResearch({
  repo,
  window,
  ghClient = createGhClient(),
  priorityConfig = {},
  limits = DEFAULT_LIMITS,
} = {}) {
  const repoName = normalizeRepoName(repo);
  const failures = [];
  let pullRequests = [];
  let releases = [];
  let commits = [];

  try {
    pullRequests = (await ghClient.listMergedPullRequests(
      repoName,
      window,
      limits.pull_requests,
    )).map((pr) => normalizePullRequest(pr, repoName, priorityConfig))
      .sort((left, right) => right.priority_score - left.priority_score);
  } catch (error) {
    failures.push({ source: 'pull_requests', message: error.message });
  }

  try {
    releases = (await ghClient.listReleases(
      repoName,
      limits.releases,
    )).map((release) => normalizeRelease(release, repoName));
  } catch (error) {
    failures.push({ source: 'releases', message: error.message });
  }

  try {
    commits = (await ghClient.listCommits(
      repoName,
      window,
      limits.commits,
    )).map((commit) => normalizeCommit(commit, repoName));
  } catch (error) {
    failures.push({ source: 'commits', message: error.message });
  }

  return {
    repo: repoName,
    pull_requests: pullRequests,
    releases,
    commits,
    failures,
  };
}

module.exports = {
  DEFAULT_LIMITS,
  DEFAULT_PRIORITY_WEIGHTS,
  createGhClient,
  collectRepoResearch,
  detectSignals,
  normalizeRepoName,
  rankCandidatePriority,
  assertNonEmptyResearch,
};
