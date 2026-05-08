(function () {
  const btn = document.getElementById('bibtex-copy');
  const code = document.getElementById('bibtex-code');
  if (!btn || !code) return;

  btn.addEventListener('click', async () => {
    const text = code.textContent.trim();
    const labelSpan = btn.querySelector('span:last-child');
    const orig = labelSpan ? labelSpan.textContent : btn.textContent;
    try {
      await navigator.clipboard.writeText(text);
      if (labelSpan) labelSpan.textContent = 'Copied'; else btn.textContent = 'Copied';
      btn.classList.add('is-success');
      setTimeout(() => {
        if (labelSpan) labelSpan.textContent = orig; else btn.textContent = orig;
        btn.classList.remove('is-success');
      }, 1400);
    } catch (e) {
      const range = document.createRange();
      range.selectNodeContents(code);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });
})();
