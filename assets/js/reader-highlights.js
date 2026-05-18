(function () {
  var body = document.querySelector('[data-reader-highlights="true"]');
  if (!body || !window.getSelection) return;

  var HIGHLIGHT_NAME = 'mf-reader-highlights';
  var key = 'mf:reader-highlights:' + (body.getAttribute('data-reader-highlights-key') || location.pathname);
  var tools = document.getElementById('post-reader-tools');
  var countEl = document.getElementById('reader-highlights-count');
  var clearBtn = document.getElementById('reader-highlights-clear');
  var listEl = document.getElementById('reader-highlights-list');
  var activeRange = null;
  var activeHighlight = null;
  var hideTimer = 0;

  var popover = document.createElement('div');
  popover.className = 'reader-highlight-popover';
  popover.hidden = true;
  popover.innerHTML = [
    '<button type="button" data-reader-action="highlight">划线</button>',
    '<button type="button" data-reader-action="copy">复制引用</button>',
    '<button type="button" data-reader-action="delete" hidden>删除划线</button>',
    '<span class="reader-highlight-status" aria-live="polite"></span>'
  ].join('');
  document.body.appendChild(popover);

  var statusEl = popover.querySelector('.reader-highlight-status');
  var deleteSelectionBtn = popover.querySelector('[data-reader-action="delete"]');

  function getHighlights() {
    try {
      var value = JSON.parse(localStorage.getItem(key) || '[]');
      return Array.isArray(value) ? value : [];
    } catch (error) {
      return [];
    }
  }

  function setHighlights(items) {
    if (!items.length) {
      localStorage.removeItem(key);
      return;
    }
    localStorage.setItem(key, JSON.stringify(items.slice(0, 200)));
  }

  function textNodes() {
    var nodes = [];
    var walker = document.createTreeWalker(
      body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode: function (node) {
          if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
          var parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          if (parent.closest('script, style, .reader-highlight-popover')) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );
    while (walker.nextNode()) nodes.push(walker.currentNode);
    return nodes;
  }

  function bodyText(nodes) {
    return nodes.map(function (node) { return node.nodeValue; }).join('');
  }

  function offsetsForRange(range, nodes) {
    var start = -1;
    var end = -1;
    var index = 0;

    for (var i = 0; i < nodes.length; i += 1) {
      var node = nodes[i];
      if (node === range.startContainer) start = index + range.startOffset;
      if (node === range.endContainer) end = index + range.endOffset;
      index += node.nodeValue.length;
    }

    if (start < 0 || end < 0 || end <= start) return null;
    return { start: start, end: end };
  }

  function rangeFromOffsets(start, end, nodes) {
    var range = document.createRange();
    var index = 0;
    var foundStart = false;
    var foundEnd = false;

    for (var i = 0; i < nodes.length; i += 1) {
      var node = nodes[i];
      var next = index + node.nodeValue.length;

      if (!foundStart && start <= next) {
        range.setStart(node, Math.max(0, start - index));
        foundStart = true;
      }
      if (!foundEnd && end <= next) {
        range.setEnd(node, Math.max(0, end - index));
        foundEnd = true;
        break;
      }
      index = next;
    }

    return foundStart && foundEnd && !range.collapsed ? range : null;
  }

  function selectorFromRange(range) {
    var nodes = textNodes();
    var offsets = offsetsForRange(range, nodes);
    if (!offsets) return null;

    var text = bodyText(nodes);
    var exact = text.slice(offsets.start, offsets.end);
    if (!exact.trim()) return null;

    return {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      exact: exact,
      prefix: text.slice(Math.max(0, offsets.start - 48), offsets.start),
      suffix: text.slice(offsets.end, offsets.end + 48),
      start: offsets.start,
      end: offsets.end,
      createdAt: new Date().toISOString()
    };
  }

  function locateSelector(selector, nodes, text) {
    if (
      Number.isInteger(selector.start) &&
      Number.isInteger(selector.end) &&
      text.slice(selector.start, selector.end) === selector.exact
    ) {
      return rangeFromOffsets(selector.start, selector.end, nodes);
    }

    var index = -1;
    var from = 0;
    while ((index = text.indexOf(selector.exact, from)) !== -1) {
      var before = text.slice(Math.max(0, index - selector.prefix.length), index);
      var after = text.slice(index + selector.exact.length, index + selector.exact.length + selector.suffix.length);
      if ((!selector.prefix || before === selector.prefix) && (!selector.suffix || after === selector.suffix)) {
        return rangeFromOffsets(index, index + selector.exact.length, nodes);
      }
      from = index + Math.max(1, selector.exact.length);
    }

    index = text.indexOf(selector.exact);
    return index >= 0 ? rangeFromOffsets(index, index + selector.exact.length, nodes) : null;
  }

  function clearDomMarks() {
    body.querySelectorAll('mark[data-reader-highlight]').forEach(function (mark) {
      var parent = mark.parentNode;
      while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
      parent.removeChild(mark);
      parent.normalize();
    });
  }

  function renderDomFallback(restored) {
    clearDomMarks();
    restored
      .slice()
      .sort(function (a, b) { return b.start - a.start; })
      .forEach(function (item) {
        var mark = document.createElement('mark');
        mark.className = 'reader-highlight-mark';
        mark.setAttribute('data-reader-highlight', 'true');
        try {
          mark.appendChild(item.range.extractContents());
          item.range.insertNode(mark);
        } catch (error) {
          mark.remove();
        }
      });
  }

  function renderHighlightList(saved) {
    if (!listEl) return;

    listEl.textContent = '';
    saved.forEach(function (item, index) {
      var li = document.createElement('li');
      li.className = 'reader-highlights-item';

      var quote = document.createElement('span');
      quote.className = 'reader-highlights-quote';
      quote.textContent = item.exact.replace(/\s+/g, ' ').trim();

      var button = document.createElement('button');
      button.type = 'button';
      button.textContent = '删除';
      button.setAttribute('data-reader-delete-highlight', item.id || String(index));
      button.setAttribute('data-reader-delete-index', String(index));

      li.appendChild(quote);
      li.appendChild(button);
      listEl.appendChild(li);
    });
  }

  function renderHighlights() {
    if (!('CSS' in window) || !CSS.highlights || !window.Highlight) clearDomMarks();

    var saved = getHighlights();
    var nodes = textNodes();
    var text = bodyText(nodes);
    var restored = saved.map(function (selector) {
      var range = locateSelector(selector, nodes, text);
      return range ? { range: range, start: selector.start } : null;
    }).filter(Boolean);

    if ('CSS' in window && CSS.highlights && window.Highlight) {
      CSS.highlights.delete(HIGHLIGHT_NAME);
      if (restored.length) {
        CSS.highlights.set(HIGHLIGHT_NAME, new Highlight(...restored.map(function (item) {
          return item.range;
        })));
      }
    } else {
      renderDomFallback(restored);
    }

    if (tools && countEl) {
      tools.hidden = saved.length === 0;
      countEl.textContent = saved.length ? '本页 ' + saved.length + ' 处划线，仅保存在当前浏览器' : '本页暂无划线';
    }
    renderHighlightList(saved);
  }

  function selectedRange() {
    var selection = window.getSelection();
    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;

    var range = selection.getRangeAt(0);
    var ancestor = range.commonAncestorContainer.nodeType === Node.TEXT_NODE
      ? range.commonAncestorContainer.parentNode
      : range.commonAncestorContainer;

    if (!body.contains(ancestor)) return null;
    if (!range.toString().trim()) return null;
    return range.cloneRange();
  }

  function findHighlightForRange(range) {
    var nodes = textNodes();
    var offsets = offsetsForRange(range, nodes);
    if (!offsets) return null;

    var text = bodyText(nodes);
    var saved = getHighlights();

    for (var i = 0; i < saved.length; i += 1) {
      var item = saved[i];
      var located = locateSelector(item, nodes, text);
      var itemOffsets = located ? offsetsForRange(located, nodes) : null;
      if (!itemOffsets) continue;

      var overlaps = offsets.start < itemOffsets.end && offsets.end > itemOffsets.start;
      if (overlaps) {
        return {
          id: item.id || String(i),
          index: i
        };
      }
    }

    return null;
  }

  function placePopover(range) {
    var rect = range.getBoundingClientRect();
    if ((!rect || !rect.width) && range.getClientRects().length) {
      rect = range.getClientRects()[0];
    }
    if (!rect) return;

    activeRange = range;
    activeHighlight = findHighlightForRange(range);
    if (deleteSelectionBtn) deleteSelectionBtn.hidden = !activeHighlight;
    popover.hidden = false;
    statusEl.textContent = '';

    var top = window.scrollY + rect.top - popover.offsetHeight - 10;
    var left = window.scrollX + rect.left + rect.width / 2 - popover.offsetWidth / 2;
    left = Math.max(12, Math.min(left, window.scrollX + document.documentElement.clientWidth - popover.offsetWidth - 12));

    if (top < window.scrollY + 8) top = window.scrollY + rect.bottom + 10;
    popover.style.top = top + 'px';
    popover.style.left = left + 'px';
  }

  function showPopoverSoon(delay) {
    window.clearTimeout(hideTimer);
    window.setTimeout(function () {
      var range = selectedRange();
      if (range) placePopover(range);
    }, delay || 0);
  }

  function hidePopoverSoon() {
    window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(function () {
      popover.hidden = true;
      activeRange = null;
      activeHighlight = null;
    }, 120);
  }

  function addHighlight() {
    if (!activeRange) return;
    var selector = selectorFromRange(activeRange);
    if (!selector) return;

    var saved = getHighlights();
    var duplicate = saved.some(function (item) {
      return item.exact === selector.exact && item.start === selector.start && item.end === selector.end;
    });
    if (!duplicate) {
      saved.push(selector);
      setHighlights(saved);
      renderHighlights();
    }
    statusEl.textContent = duplicate ? '已存在' : '已划线';
    window.getSelection().removeAllRanges();
    hidePopoverSoon();
  }

  function textFragmentUrl(text) {
    var exact = text.replace(/\s+/g, ' ').trim();
    if (exact.length > 180) exact = exact.slice(0, 180);
    return location.origin + location.pathname + location.search + '#:~:text=' + encodeURIComponent(exact);
  }

  function copyText(value) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(value);
    }

    var field = document.createElement('textarea');
    field.value = value;
    field.setAttribute('readonly', '');
    field.style.position = 'fixed';
    field.style.left = '-9999px';
    document.body.appendChild(field);
    field.select();
    document.execCommand('copy');
    field.remove();
    return Promise.resolve();
  }

  function copyReference() {
    if (!activeRange) return;
    var value = textFragmentUrl(activeRange.toString());
    copyText(value).then(function () {
      statusEl.textContent = '已复制';
      hidePopoverSoon();
    }).catch(function () {
      statusEl.textContent = '复制失败';
    });
  }

  function deleteActiveHighlight() {
    if (!activeHighlight) return;

    deleteHighlight(activeHighlight.id, activeHighlight.index);
    statusEl.textContent = '已删除';
    window.getSelection().removeAllRanges();
    hidePopoverSoon();
  }

  function deleteHighlight(identifier, fallbackIndex) {
    var saved = getHighlights();
    var next = saved.filter(function (item, index) {
      if (item.id) return item.id !== identifier;
      return index !== fallbackIndex;
    });

    setHighlights(next);
    if ('CSS' in window && CSS.highlights) CSS.highlights.delete(HIGHLIGHT_NAME);
    clearDomMarks();
    renderHighlights();
  }

  popover.addEventListener('mousedown', function (event) {
    event.preventDefault();
  });

  popover.addEventListener('click', function (event) {
    var button = event.target.closest('button[data-reader-action]');
    if (!button) return;
    if (button.getAttribute('data-reader-action') === 'highlight') addHighlight();
    if (button.getAttribute('data-reader-action') === 'copy') copyReference();
    if (button.getAttribute('data-reader-action') === 'delete') deleteActiveHighlight();
  });

  body.addEventListener('mouseup', function () { showPopoverSoon(0); });
  body.addEventListener('keyup', function () { showPopoverSoon(0); });
  body.addEventListener('touchend', function () { showPopoverSoon(320); });

  document.addEventListener('mousedown', function (event) {
    if (!popover.contains(event.target)) hidePopoverSoon();
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') hidePopoverSoon();
  });
  window.addEventListener('scroll', function () {
    if (!popover.hidden) hidePopoverSoon();
  }, { passive: true });

  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      setHighlights([]);
      if ('CSS' in window && CSS.highlights) CSS.highlights.delete(HIGHLIGHT_NAME);
      clearDomMarks();
      renderHighlights();
    });
  }

  if (tools) {
    tools.addEventListener('click', function (event) {
      var button = event.target.closest('button[data-reader-delete-highlight]');
      if (!button) return;

      var index = parseInt(button.getAttribute('data-reader-delete-index') || '-1', 10);
      deleteHighlight(button.getAttribute('data-reader-delete-highlight'), index);
    });
  }

  renderHighlights();
})();
