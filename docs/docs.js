// Minimal Python highlighter for <pre><code> blocks + active-nav marking.
// Self-contained — no external dependencies.
(function () {
  var TOKEN = new RegExp(
    [
      "(#.*$)",                                              // 1 comment
      "(\"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)*')",    // 2 string
      "\\b(def|class|return|import|from|as|for|while|if|elif|else|with|in|not|and|or|lambda|None|True|False|assert|raise|pass|yield)\\b", // 3 keyword
      "\\b(\\d+(?:\\.\\d+)?(?:e-?\\d+)?)\\b",                // 4 number
    ].join("|"),
    "gm"
  );

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlight(text) {
    return escapeHtml(text)
      .split("\n")
      .map(function (line) {
        if (/^\s*⇒/.test(line)) {
          return '<span class="tok-out">' + line + "</span>";
        }
        return line.replace(TOKEN, function (m, com, str, kw, num) {
          if (com) return '<span class="tok-com">' + com + "</span>";
          if (str) return '<span class="tok-str">' + str + "</span>";
          if (kw) return '<span class="tok-kw">' + kw + "</span>";
          return '<span class="tok-num">' + num + "</span>";
        });
      })
      .join("\n");
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("pre > code").forEach(function (el) {
      el.innerHTML = highlight(el.textContent);
    });
    var here = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".sidebar nav a").forEach(function (a) {
      var href = (a.getAttribute("href") || "").split("#")[0];
      if (href === here) a.classList.add("active");
    });
  });
})();
