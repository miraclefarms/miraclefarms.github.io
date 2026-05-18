(function () {
  var comments = document.getElementById('post-comments');
  if (!comments) return;

  var mount = comments.querySelector('[data-giscus-mount]');
  if (!mount || mount.dataset.loaded === 'true') return;

  function config(name, fallback) {
    var value = comments.getAttribute('data-giscus-' + name);
    return value || fallback || '';
  }

  function loadGiscus() {
    if (mount.dataset.loaded === 'true') return;
    mount.dataset.loaded = 'true';
    mount.textContent = '';

    var script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.setAttribute('data-repo', config('repo'));
    script.setAttribute('data-repo-id', config('repo-id'));
    script.setAttribute('data-category', config('category'));
    script.setAttribute('data-category-id', config('category-id'));
    script.setAttribute('data-mapping', config('mapping', 'pathname'));
    script.setAttribute('data-strict', config('strict', '0'));
    script.setAttribute('data-reactions-enabled', config('reactions-enabled', '1'));
    script.setAttribute('data-emit-metadata', config('emit-metadata', '0'));
    script.setAttribute('data-input-position', config('input-position', 'bottom'));
    script.setAttribute('data-theme', config('theme', 'noborder_light'));
    script.setAttribute('data-lang', config('lang', document.documentElement.lang || 'zh-CN'));

    mount.appendChild(script);
  }

  if (!('IntersectionObserver' in window)) {
    loadGiscus();
    return;
  }

  var observer = new IntersectionObserver(function (entries) {
    if (entries.some(function (entry) { return entry.isIntersecting; })) {
      observer.disconnect();
      loadGiscus();
    }
  }, { rootMargin: '480px 0px' });

  observer.observe(comments);
})();
