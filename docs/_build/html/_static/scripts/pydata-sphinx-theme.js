(() => {
  "use strict";
  function e(e) {
    "loading" != document.readyState
      ? e()
      : document.addEventListener("DOMContentLoaded", e);
  }
  const t = (e) => "string" == typeof e && /^[v\d]/.test(e) && o.test(e),
    n = (e, t, n) => {
      u(n);
      const o = ((e, t) => {
        const n = r(e),
          o = r(t),
          a = n.pop(),
          c = o.pop(),
          s = i(n, o);
        return 0 !== s
          ? s
          : a && c
            ? i(a.split("."), c.split("."))
            : a || c
              ? a
                ? -1
                : 1
              : 0;
      })(e, t);
      return d[n].includes(o);
    },
    o =
      /^[v^~<>=]*?(\d+)(?:\.([x*]|\d+)(?:\.([x*]|\d+)(?:\.([x*]|\d+))?(?:-([\da-z\-]+(?:\.[\da-z\-]+)*))?(?:\+[\da-z\-]+(?:\.[\da-z\-]+)*)?)?)?$/i,
    r = (e) => {
      if ("string" != typeof e)
        throw new TypeError("Invalid argument expected string");
      const t = e.match(o);
      if (!t)
        throw new Error(`Invalid argument not valid semver ('${e}' received)`);
      return t.shift(), t;
    },
    a = (e) => "*" === e || "x" === e || "X" === e,
    c = (e) => {
      const t = parseInt(e, 10);
      return isNaN(t) ? e : t;
    },
    s = (e, t) => {
      if (a(e) || a(t)) return 0;
      const [n, o] = ((e, t) =>
        typeof e != typeof t ? [String(e), String(t)] : [e, t])(c(e), c(t));
      return n > o ? 1 : n < o ? -1 : 0;
    },
    i = (e, t) => {
      for (let n = 0; n < Math.max(e.length, t.length); n++) {
        const o = s(e[n] || "0", t[n] || "0");
        if (0 !== o) return o;
      }
      return 0;
    },
    d = { ">": [1], ">=": [0, 1], "=": [0], "<=": [-1, 0], "<": [-1] },
    l = Object.keys(d),
    u = (e) => {
      if ("string" != typeof e)
        throw new TypeError(
          "Invalid operator type, expected string but got " + typeof e,
        );
      if (-1 === l.indexOf(e))
        throw new Error(`Invalid operator, expected one of ${l.join("|")}`);
    };
  var m = window.matchMedia("(prefers-color-scheme: dark)");
  function h(e) {
    document.documentElement.dataset.theme = m.matches ? "dark" : "light";
  }
  function p(e) {
    "light" !== e &&
      "dark" !== e &&
      "auto" !== e &&
      (console.error(`Got invalid theme mode: ${e}. Resetting to auto.`),
      (e = "auto"));
    var t = m.matches ? "dark" : "light";
    document.documentElement.dataset.mode = e;
    var n = "auto" == e ? t : e;
    (document.documentElement.dataset.theme = n),
      document.querySelectorAll(".dropdown-menu").forEach((e) => {
        "dark" === n
          ? e.classList.add("dropdown-menu-dark")
          : e.classList.remove("dropdown-menu-dark");
      }),
      localStorage.setItem("mode", e),
      localStorage.setItem("theme", n),
      console.log(`[PST]: Changed to ${e} mode using the ${n} theme.`),
      (m.onchange = "auto" == e ? h : "");
  }
  function f() {
    const e = document.documentElement.dataset.defaultMode || "auto",
      t = localStorage.getItem("mode") || e;
    var n, o;
    p(
      ((o =
        (n = m.matches
          ? ["auto", "light", "dark"]
          : ["auto", "dark", "light"]).indexOf(t) + 1) === n.length && (o = 0),
      n[o]),
    );
  }
  var g = () => {
      let e = document.querySelectorAll("form.bd-search");
      return e.length
        ? (1 == e.length
            ? e[0]
            : document.querySelector(
                "div:not(.search-button__search-container) > form.bd-search",
              )
          ).querySelector("input")
        : void 0;
    },
    v = () => {
      let e = g(),
        t = document.querySelector(".search-button__wrapper");
      e === t.querySelector("input") && t.classList.toggle("show"),
        document.activeElement === e
          ? e.blur()
          : (e.focus(), e.select(), e.scrollIntoView({ block: "center" }));
    },
    y =
      0 === navigator.platform.indexOf("Mac") ||
      "iPhone" === navigator.platform,
    b = () =>
      "dirhtml" == DOCUMENTATION_OPTIONS.BUILDER
        ? "index" == DOCUMENTATION_OPTIONS.pagename
          ? "/"
          : `${DOCUMENTATION_OPTIONS.pagename}/`
        : `${DOCUMENTATION_OPTIONS.pagename}.html`;
  async function w(e) {
    document.querySelector("#bd-header-version-warning").remove();
    const t = DOCUMENTATION_OPTIONS.VERSION,
      n = new Date(),
      o = JSON.parse(localStorage.getItem("pst_banner_pref") || "{}");
    console.debug(
      `[PST] Dismissing the version warning banner on ${t} starting ${n}.`,
    ),
      (o[t] = n),
      localStorage.setItem("pst_banner_pref", JSON.stringify(o));
  }
  async function E(e) {
    e.preventDefault();
    const t = b();
    let n = e.currentTarget.getAttribute("href"),
      o = n.replace(t, "");
    try {
      (await fetch(n, { method: "HEAD" })).ok
        ? (location.href = n)
        : (location.href = o);
    } catch (e) {
      location.href = o;
    }
  }
  async function S() {
    var e = document.querySelectorAll(".version-switcher__button");
    const o = e.length > 0,
      r = DOCUMENTATION_OPTIONS.hasOwnProperty("theme_switcher_json_url"),
      a = DOCUMENTATION_OPTIONS.show_version_warning_banner;
    if (r && (o || a)) {
      const o = await (async function (e) {
        try {
          var t = new URL(e);
        } catch (n) {
          if (!(n instanceof TypeError)) throw n;
          {
            if (!window.location.origin) return null;
            const n = await fetch(window.location.origin, { method: "HEAD" });
            t = new URL(e, n.url);
          }
        }
        const n = await fetch(t);
        return await n.json();
      })(DOCUMENTATION_OPTIONS.theme_switcher_json_url);
      o &&
        ((function (e, t) {
          const n = b();
          t.forEach((e) => {
            (e.dataset.activeVersionName = ""), (e.dataset.activeVersion = "");
          });
          const o = (e = e.map(
            (e) => (
              (e.match =
                e.version ==
                DOCUMENTATION_OPTIONS.theme_switcher_version_match),
              (e.preferred = e.preferred || !1),
              "name" in e || (e.name = e.version),
              e
            ),
          ))
            .map((e) => e.preferred && e.match)
            .some(Boolean);
          var r = !1;
          e.forEach((e) => {
            const a = document.createElement("a");
            a.setAttribute(
              "class",
              "dropdown-item list-group-item list-group-item-action py-1",
            ),
              a.setAttribute("href", `${e.url}${n}`),
              a.setAttribute("role", "option");
            const c = document.createElement("span");
            (c.textContent = `${e.name}`),
              a.appendChild(c),
              (a.dataset.versionName = e.name),
              (a.dataset.version = e.version);
            let s = o && e.preferred,
              i = !o && !r && e.match;
            (s || i) &&
              (a.classList.add("active"),
              t.forEach((t) => {
                (t.innerText = e.name),
                  (t.dataset.activeVersionName = e.name),
                  (t.dataset.activeVersion = e.version);
              }),
              (r = !0)),
              document
                .querySelectorAll(".version-switcher__menu")
                .forEach((e) => {
                  let t = a.cloneNode(!0);
                  (t.onclick = E), e.append(t);
                });
          });
        })(o, e),
        a &&
          (function (e) {
            var o = DOCUMENTATION_OPTIONS.VERSION,
              r = e.filter((e) => e.preferred);
            if (1 !== r.length) {
              const e = 0 == r.length ? "No" : "Multiple";
              return void console.log(
                `[PST] ${e} versions marked "preferred" found in versions JSON, ignoring.`,
              );
            }
            const a = r[0].version,
              c = r[0].url,
              s = t(o) && t(a);
            if (s && n(o, a, "="))
              return void console.log(
                "This is the prefered version of the docs, not showing the warning banner.",
              );
            const i = JSON.parse(
              localStorage.getItem("pst_banner_pref") || "{}",
            )[o];
            if (null != i) {
              const e = new Date(i),
                t = (new Date() - e) / 864e5;
              if (t < 14)
                return void console.info(
                  `[PST] Suppressing version warning banner; was dismissed ${Math.floor(t)} day(s) ago`,
                );
            }
            const d = document.querySelector("#bd-header-version-warning"),
              l = document.createElement("div"),
              u = document.createElement("div"),
              m = document.createElement("strong"),
              h = document.createElement("a"),
              p = document.createElement("a");
            (l.classList = "bd-header-announcement__content  ms-auto me-auto"),
              (u.classList = "sidebar-message"),
              (h.classList =
                "btn text-wrap font-weight-bold ms-3 my-1 align-baseline pst-button-link-to-stable-version"),
              (h.href = `${c}${b()}`),
              (h.innerText = "Switch to stable version"),
              (h.onclick = E),
              (p.classList = "ms-3 my-1 align-baseline");
            const f = document.createElement("i");
            p.append(f),
              (f.classList = "fa-solid fa-xmark"),
              (p.onclick = w),
              (u.innerText = "This is documentation for ");
            const g =
                o.includes("dev") || o.includes("rc") || o.includes("pre"),
              v = s && n(o, a, ">");
            g || v
              ? (m.innerText = "an unstable development version")
              : s && n(o, a, "<")
                ? (m.innerText = `an old version (${o})`)
                : (m.innerText = o ? `version ${o}` : "an unknown version"),
              d.appendChild(l),
              d.append(p),
              l.appendChild(u),
              u.appendChild(m),
              u.appendChild(document.createTextNode(".")),
              u.appendChild(h),
              d.classList.remove("d-none");
          })(o));
    }
  }
  function T() {
    const e = () => {
        document
          .querySelectorAll(
            "pre, .nboutput > .output_area, .cell_output > .output, .jp-RenderedHTMLCommon",
          )
          .forEach((e) => {
            e.tabIndex =
              e.scrollWidth > e.clientWidth || e.scrollHeight > e.clientHeight
                ? 0
                : -1;
          });
      },
      t = (function (e, t) {
        let n = null;
        return (...t) => {
          clearTimeout(n),
            (n = setTimeout(() => {
              e(...t);
            }, 300));
        };
      })(e);
    window.addEventListener("resize", t),
      new MutationObserver(t).observe(document.getElementById("main-content"), {
        subtree: !0,
        childList: !0,
      }),
      e();
  }
  async function O() {
    const e = document.querySelector(".bd-header-announcement"),
      { pstAnnouncementUrl: t } = e ? e.dataset : null;
    if (t)
      try {
        const n = await fetch(t);
        if (!n.ok)
          throw new Error(
            `[PST]: HTTP response status not ok: ${n.status} ${n.statusText}`,
          );
        const o = await n.text();
        if (0 === o.length)
          return void console.log(`[PST]: Empty announcement at: ${t}`);
        (e.innerHTML = `<div class="bd-header-announcement__content">${o}</div>`),
          e.classList.remove("d-none");
      } catch (e) {
        console.log(`[PST]: Failed to load announcement at: ${t}`),
          console.error(e);
      }
  }
  e(async function () {
    await Promise.allSettled([S(), O()]);
    const e = document.querySelector(".pst-async-banner-revealer");
    if (!e) return;
    e.classList.remove("d-none");
    const t = Array.from(e.children).reduce((e, t) => e + t.offsetHeight, 0);
    e.style.setProperty("height", `${t}px`),
      setTimeout(() => {
        e.style.setProperty("height", "auto");
      }, 320);
  }),
    e(function () {
      p(document.documentElement.dataset.mode),
        document.querySelectorAll(".theme-switch-button").forEach((e) => {
          e.addEventListener("click", f);
        });
    }),
    e(function () {
      if (!document.querySelector(".bd-docs-nav")) return;
      var e = document.querySelector("div.bd-sidebar");
      let t = parseInt(sessionStorage.getItem("sidebar-scroll-top"), 10);
      if (isNaN(t)) {
        var n = document
          .querySelector(".bd-docs-nav")
          .querySelectorAll(".active");
        if (n.length > 0) {
          var o = n[n.length - 1],
            r = o.getBoundingClientRect().y - e.getBoundingClientRect().y;
          if (o.getBoundingClientRect().y > 0.5 * window.innerHeight) {
            let t = 0.25;
            (e.scrollTop = r - e.clientHeight * t),
              console.log("[PST]: Scrolled sidebar using last active link...");
          }
        }
      } else
        (e.scrollTop = t),
          console.log(
            "[PST]: Scrolled sidebar using stored browser position...",
          );
      window.addEventListener("beforeunload", () => {
        sessionStorage.setItem("sidebar-scroll-top", e.scrollTop);
      });
    }),
    e(function () {
      window.addEventListener("activate.bs.scrollspy", function () {
        document.querySelectorAll(".bd-toc-nav a").forEach((e) => {
          e.parentElement.classList.remove("active");
        }),
          document.querySelectorAll(".bd-toc-nav a.active").forEach((e) => {
            e.parentElement.classList.add("active");
          });
      });
    }),
    e(() => {
      (() => {
        let e = document.querySelectorAll(".search-button__kbd-shortcut");
        y &&
          e.forEach(
            (e) =>
              (e.querySelector("kbd.kbd-shortcut__modifier").innerText = "⌘"),
          );
      })(),
        window.addEventListener(
          "keydown",
          (e) => {
            let t = g();
            e.shiftKey ||
            e.altKey ||
            (y ? !e.metaKey || e.ctrlKey : e.metaKey || !e.ctrlKey) ||
            !/^k$/i.test(e.key)
              ? document.activeElement === t && /Escape/i.test(e.key) && v()
              : (e.preventDefault(), v());
          },
          !0,
        ),
        document.querySelectorAll(".search-button__button").forEach((e) => {
          e.onclick = v;
        });
      let e = document.querySelector(".search-button__overlay");
      e && (e.onclick = v);
    }),
    e(function () {
      new MutationObserver((e, t) => {
        e.forEach((e) => {
          0 !== e.addedNodes.length &&
            void 0 !== e.addedNodes[0].data &&
            -1 != e.addedNodes[0].data.search("Inserted RTD Footer") &&
            e.addedNodes.forEach((e) => {
              document.getElementById("rtd-footer-container").append(e);
            });
        });
      }).observe(document.body, { childList: !0 });
    }),
    e(function () {
      const e = document.getElementById("pst-primary-sidebar-checkbox"),
        t = document.getElementById("pst-secondary-sidebar-checkbox"),
        n = document.querySelector(".bd-sidebar-primary"),
        o = document.querySelector(".bd-sidebar-secondary"),
        r = document.querySelector(".primary-toggle"),
        a = document.querySelector(".secondary-toggle");
      [
        [r, e, n],
        [a, t, o],
      ].forEach(([e, t, n]) => {
        e &&
          e.addEventListener("click", (e) => {
            if (
              (e.preventDefault(),
              e.stopPropagation(),
              (t.checked = !t.checked),
              t.checked)
            ) {
              const e = n.querySelector("a, button");
              setTimeout(() => e.focus(), 100);
            }
          });
      }),
        [
          [n, e, r],
          [o, t, a],
        ].forEach(([e, t, n]) => {
          e &&
            e.addEventListener("keydown", (e) => {
              "Escape" === e.key &&
                (e.preventDefault(),
                e.stopPropagation(),
                (t.checked = !1),
                n.focus());
            });
        }),
        [
          [e, r],
          [t, a],
        ].forEach(([e, t]) => {
          e.addEventListener("change", (e) => {
            e.currentTarget.checked || t.focus();
          });
        });
    }),
    "complete" === document.readyState
      ? T()
      : window.addEventListener("load", T);
})();
//# sourceMappingURL=pydata-sphinx-theme.js.map