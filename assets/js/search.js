(function () {
  var root = document.querySelector('[data-search-page]');
  if (!root) return;

  var form = root.querySelector('[data-search-form]');
  var input = root.querySelector('[data-search-input]');
  var status = root.querySelector('[data-search-status]');
  var resultsEl = root.querySelector('[data-search-results]');
  var modeButtons = Array.from(root.querySelectorAll('[data-search-mode]'));
  var filterButtons = Array.from(root.querySelectorAll('[data-kind]'));
  var filtersEl = root.querySelector('[data-search-filters]');

  var state = {
    mode: 'pagefind',
    kind: 'all',
    pagefind: null,
    loadError: null,
    runId: 0
  };

  var labels = {
    brief: 'Brief',
    reading: 'Reading',
    'field-note': 'Field Note',
    essay: 'Essay',
    'founding-note': 'Founding Note'
  };

  function googleUrl(query) {
    return 'https://www.google.com/search?q=' + encodeURIComponent('site:miraclefarms.github.io ' + query);
  }

  function setStatus(text) {
    status.textContent = text;
  }

  function clearResults() {
    resultsEl.replaceChildren();
  }

  function updateUrl(query) {
    var params = new URLSearchParams();
    if (query) params.set('q', query);
    if (state.mode !== 'pagefind') params.set('mode', state.mode);
    if (state.kind !== 'all') params.set('kind', state.kind);
    var next = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
    window.history.replaceState(null, '', next);
  }

  function setMode(mode) {
    state.mode = mode === 'google' ? 'google' : 'pagefind';
    modeButtons.forEach(function (button) {
      var active = button.getAttribute('data-search-mode') === state.mode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    if (filtersEl) filtersEl.hidden = state.mode === 'google';
  }

  function setKind(kind) {
    state.kind = kind || 'all';
    filterButtons.forEach(function (button) {
      button.classList.toggle('is-active', button.getAttribute('data-kind') === state.kind);
    });
  }

  function makeEl(tag, className, text) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    if (typeof text === 'string') el.textContent = text;
    return el;
  }

  function truncateText(text, limit) {
    var normalized = (text || '').replace(/\s+/g, ' ').trim();
    if (normalized.length <= limit) return normalized;
    return normalized.slice(0, limit - 1).trim() + '...';
  }

  function renderGoogle(query, message) {
    clearResults();
    if (!query) {
      setStatus('输入关键词开始搜索。');
      return;
    }

    setStatus(message || '使用 Google 搜索 miraclefarms.github.io。');

    var item = makeEl('li', 'search-result search-google-result');
    var title = makeEl('h2', null, 'Google site search');
    var text = makeEl('p', null, 'site:miraclefarms.github.io ' + query);
    var link = makeEl('a', 'search-google-link');
    link.href = googleUrl(query);
    link.rel = 'noopener noreferrer';
    link.textContent = 'Open Google results';
    link.target = '_blank';
    link.setAttribute('aria-label', 'Open Google results for ' + query);
    item.append(title, text, link);
    resultsEl.append(item);
  }

  function normalizeUrl(url) {
    if (!url) return '/';
    return url.replace(/\/index\.html$/, '/');
  }

  function renderPagefindResults(items, query) {
    clearResults();

    if (!items.length) {
      renderGoogle(query, 'Pagefind 没有找到结果。');
      return;
    }

    setStatus(items.length + ' results from Pagefind');

    items.forEach(function (data) {
      var meta = data.meta || {};
      var firstSubResult = data.sub_results && data.sub_results[0];
      var kind = meta.kind || '';
      var item = makeEl('li', 'search-result');
      var head = makeEl('div', 'search-result-head');
      var chip = makeEl('span', 'search-result-kind', labels[kind] || meta.category || 'Post');
      var date = makeEl('span', 'search-result-date', meta.date || '');
      var link = makeEl('a', 'search-result-title', meta.title || data.url || 'Untitled');
      var excerptText = firstSubResult && (firstSubResult.plain_excerpt || firstSubResult.excerpt);
      var excerpt = makeEl('p', 'search-result-excerpt', excerptText || meta.intro || truncateText(data.content, 220));
      var url = normalizeUrl(data.url);

      link.href = url;
      head.append(chip);
      if (meta.date) head.append(date);
      item.append(head, link);
      if (excerpt.textContent) item.append(excerpt);
      if (meta.tags) item.append(makeEl('p', 'search-result-tags', meta.tags));
      resultsEl.append(item);
    });
  }

  function loadPagefind() {
    if (state.pagefind || state.loadError) {
      return state.pagefind ? Promise.resolve(state.pagefind) : Promise.reject(state.loadError);
    }

    return import('/pagefind/pagefind.js')
      .then(function (pagefind) {
        state.pagefind = pagefind;
        return pagefind;
      })
      .catch(function (err) {
        state.loadError = err;
        throw err;
      });
  }

  function pagefindOptions() {
    if (state.kind === 'all') return {};
    return { filters: { kind: state.kind } };
  }

  function runPagefind(query) {
    var currentRun = ++state.runId;
    if (!query) {
      clearResults();
      setStatus('输入关键词开始搜索。');
      return;
    }

    setStatus('Searching Pagefind...');
    loadPagefind()
      .then(function (pagefind) {
        return pagefind.search(query, pagefindOptions());
      })
      .then(function (search) {
        if (currentRun !== state.runId) return;
        var resultPromises = search.results.slice(0, 20).map(function (result) {
          return result.data();
        });
        return Promise.all(resultPromises);
      })
      .then(function (items) {
        if (!items || currentRun !== state.runId) return;
        renderPagefindResults(items, query);
      })
      .catch(function () {
        if (currentRun !== state.runId) return;
        renderGoogle(query, 'Pagefind 索引尚不可用。');
      });
  }

  function runSearch() {
    var query = input.value.trim();
    updateUrl(query);

    if (state.mode === 'google') {
      if (query) window.location.href = googleUrl(query);
      return;
    }

    runPagefind(query);
  }

  modeButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      setMode(button.getAttribute('data-search-mode'));
      var query = input.value.trim();
      updateUrl(query);
      if (state.mode === 'google') {
        renderGoogle(query);
      } else {
        runPagefind(query);
      }
    });
  });

  filterButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      setKind(button.getAttribute('data-kind'));
      runSearch();
    });
  });

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    runSearch();
  });

  input.addEventListener('input', function () {
    window.clearTimeout(input.__searchTimer);
    input.__searchTimer = window.setTimeout(function () {
      if (state.mode === 'pagefind') runSearch();
      else updateUrl(input.value.trim());
    }, 180);
  });

  var params = new URLSearchParams(window.location.search);
  var initialQuery = params.get('q') || '';
  var initialMode = params.get('mode') || 'pagefind';
  var initialKind = params.get('kind') || 'all';
  input.value = initialQuery;
  setMode(initialMode);
  setKind(initialKind);

  if (initialQuery) {
    if (state.mode === 'google') renderGoogle(initialQuery);
    else runPagefind(initialQuery);
  }
})();
