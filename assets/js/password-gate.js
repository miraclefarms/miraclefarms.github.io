(function () {
  'use strict';

  var SESSION_KEY = 'mf_session_pass';
  var ITERATIONS = 100000;

  function b64ToBuffer(b64) {
    var bin = atob(b64);
    var buf = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    return buf;
  }

  function deriveKey(password, salt) {
    var enc = new TextEncoder();
    return crypto.subtle.importKey('raw', enc.encode(password), 'PBKDF2', false, ['deriveKey'])
      .then(function (km) {
        return crypto.subtle.deriveKey(
          { name: 'PBKDF2', salt: salt, iterations: ITERATIONS, hash: 'SHA-256' },
          km,
          { name: 'AES-GCM', length: 256 },
          false,
          ['decrypt']
        );
      });
  }

  function decryptBlob(password, blob) {
    var salt = b64ToBuffer(blob.salt);
    var iv   = b64ToBuffer(blob.iv);
    var ct   = b64ToBuffer(blob.ct);
    return deriveKey(password, salt).then(function (key) {
      return crypto.subtle.decrypt({ name: 'AES-GCM', iv: iv }, key, ct);
    }).then(function (plain) {
      return new TextDecoder().decode(plain);
    });
  }

  function getBlob() {
    var el = document.getElementById('__locked-blob');
    if (!el) return null;
    try { return JSON.parse(el.textContent); } catch (e) { return null; }
  }

  function injectContent(html) {
    var body = document.getElementById('post-body-content');
    if (!body) return;
    body.innerHTML = html;
    initWordCount(body);
    initToC(body);
  }

  function initWordCount(body) {
    var wcEl = document.getElementById('post-word-count');
    if (!wcEl) return;
    var text = body.innerText || '';
    var cjk  = (text.match(/[一-鿿㐀-䶿]/g) || []).length;
    var eng  = (text.replace(/[一-鿿㐀-䶿]/g, ' ').match(/\b[a-zA-Z0-9]+\b/g) || []).length;
    var mins = Math.max(1, Math.round(cjk / 400 + eng / 150));
    var total = cjk + eng;
    if (total > 0) wcEl.textContent = '约 ' + total.toLocaleString('zh-CN') + ' 字 · ' + mins + ' 分钟';
    wcEl.style.display = '';
  }

  function initToC(body) {
    var nav    = document.getElementById('post-toc-nav');
    var tocEl  = document.querySelector('.post-toc');
    if (!nav) return;
    nav.innerHTML = '';

    var headings = Array.from(body.querySelectorAll('h2'));
    if (!headings.length) return;

    if (tocEl) tocEl.style.display = '';

    var slugify = function (t) {
      return t.toLowerCase().trim()
        .replace(/[\s　]+/g, '-')
        .replace(/[^\w一-鿿\-]/g, '')
        .replace(/-+/g, '-');
    };

    headings.forEach(function (h, idx) {
      if (!h.id) h.id = slugify(h.textContent) || ('section-' + (idx + 1));
      var link = document.createElement('a');
      link.href = '#' + h.id;
      link.textContent = h.textContent;
      link.className = 'toc-link toc-h2';
      nav.appendChild(link);
    });

    var links = Array.from(nav.querySelectorAll('a'));
    var setActive = function (id) {
      links.forEach(function (l) {
        l.classList.toggle('active', l.getAttribute('href') === '#' + id);
      });
    };

    var observer = new IntersectionObserver(function (entries) {
      var visible = entries
        .filter(function (e) { return e.isIntersecting; })
        .sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
      if (visible.length) setActive(visible[0].target.id);
    }, { rootMargin: '-20% 0px -65% 0px', threshold: [0, 1] });

    headings.forEach(function (h) { observer.observe(h); });
    if (headings[0]) setActive(headings[0].id);
  }

  function tryUnlock(password) {
    var blob = getBlob();
    if (!blob) return Promise.resolve(false);
    return decryptBlob(password, blob).then(function (html) {
      injectContent(html);
      sessionStorage.setItem(SESSION_KEY, password);
      var gate = document.getElementById('password-gate');
      if (gate) gate.remove();
      return true;
    }).catch(function () {
      return false;
    });
  }

  function init() {
    var blob = getBlob();
    if (!blob) return;

    // Auto-unlock from session
    var saved = sessionStorage.getItem(SESSION_KEY);
    if (saved) {
      tryUnlock(saved).then(function (ok) {
        if (!ok) sessionStorage.removeItem(SESSION_KEY);
      });
      return;
    }

    var submitBtn = document.getElementById('pg-submit');
    var inputEl   = document.getElementById('pg-input');
    var errorEl   = document.getElementById('pg-error');
    if (!submitBtn || !inputEl) return;

    function onSubmit() {
      var pw = inputEl.value;
      if (!pw) return;
      submitBtn.disabled = true;
      submitBtn.textContent = '…';
      tryUnlock(pw).then(function (ok) {
        if (!ok) {
          if (errorEl) errorEl.hidden = false;
          submitBtn.disabled = false;
          submitBtn.textContent = '解锁';
          inputEl.value = '';
          inputEl.focus();
        }
      });
    }

    submitBtn.addEventListener('click', onSubmit);
    inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') onSubmit();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
