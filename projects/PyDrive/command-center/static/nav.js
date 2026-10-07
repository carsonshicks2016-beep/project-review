/* Two-level navigation shell.
 *
 * The dashboard grew to twelve flat tabs with heavy overlap: four of them were
 * training pipelines, three were "launch a sim", two were analysis. This groups
 * them five ways without touching app.js — the original .tab buttons and
 * #tab-<name> panels are untouched, so activateTab() and every per-tab load
 * hook still work exactly as before. All this adds is the group layer above.
 */
(function () {
  "use strict";

  var GROUP_OF = {};
  document.querySelectorAll(".subnav-rail .tabs").forEach(function (row) {
    var g = row.dataset.group;
    row.querySelectorAll(".tab").forEach(function (t) { GROUP_OF[t.dataset.tab] = g; });
  });

  var groupBtns = Array.prototype.slice.call(document.querySelectorAll(".group"));
  var subnavs = Array.prototype.slice.call(document.querySelectorAll(".subnav-rail .tabs"));

  function showGroup(g) {
    groupBtns.forEach(function (b) { b.classList.toggle("active", b.dataset.group === g); });
    subnavs.forEach(function (n) {
      var on = n.dataset.group === g;
      n.hidden = !on;
      // a one-panel group has nothing to choose between; don't show a lone tab
      if (on) n.classList.toggle("solo", n.querySelectorAll(".tab").length < 2);
    });
    document.querySelector(".subnav-rail").classList.toggle(
      "empty", document.querySelectorAll(".subnav-rail .tabs:not([hidden]).solo").length > 0);
  }

  groupBtns.forEach(function (b) {
    b.onclick = function () {
      var g = b.dataset.group;
      var row = document.querySelector('.subnav-rail .tabs[data-group="' + g + '"]');
      var current = row.querySelector('.tab.active') || row.querySelector(".tab");
      showGroup(g);
      if (current) window.activateTab(current.dataset.tab);
    };
  });

  /* Keep the group highlight in sync when something else drives the tabs —
     the Ops shortcut buttons call activateTab() directly. Function
     declarations in a classic script are aliased onto window, so wrapping the
     property also intercepts app.js's own internal calls. */
  var inner = window.activateTab;
  if (typeof inner === "function") {
    window.activateTab = function (name) {
      inner.apply(this, arguments);
      var g = GROUP_OF[name];
      if (g) showGroup(g);
    };
  }

  showGroup("ops");
})();
