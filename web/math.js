// Tiny client-side math for the Hermit web UI: safe arithmetic plus
// single-variable polynomial integrate/differentiate. No eval(), no network.

function tokenize(src) {
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === " " || c === "\t") { i++; continue; }
    if (/[0-9.]/.test(c)) {
      let j = i;
      while (j < src.length && /[0-9.]/.test(src[j])) j++;
      tokens.push({ t: "num", v: parseFloat(src.slice(i, j)) });
      i = j;
      continue;
    }
    if ("+-*/^()".includes(c)) { tokens.push({ t: c }); i++; continue; }
    return null;
  }
  return tokens;
}

function safeEval(src) {
  const tokens = tokenize(src);
  if (!tokens || tokens.length === 0) return null;
  let pos = 0;
  const peek = () => tokens[pos];
  function parseExpr() {
    let v = parseTerm();
    while (peek() && (peek().t === "+" || peek().t === "-")) {
      const op = tokens[pos++].t;
      const r = parseTerm();
      if (v === null || r === null) return null;
      v = op === "+" ? v + r : v - r;
    }
    return v;
  }
  function parseTerm() {
    let v = parseFactor();
    while (peek() && (peek().t === "*" || peek().t === "/")) {
      const op = tokens[pos++].t;
      const r = parseFactor();
      if (v === null || r === null) return null;
      if (op === "/" && r === 0) return null;
      v = op === "*" ? v * r : v / r;
    }
    return v;
  }
  function parseFactor() {
    let v = parseUnary();
    if (peek() && peek().t === "^") {
      pos++;
      const e = parseFactor();
      if (v === null || e === null) return null;
      v = Math.pow(v, e);
    }
    return v;
  }
  function parseUnary() {
    if (peek() && peek().t === "-") { pos++; const v = parseUnary(); return v === null ? null : -v; }
    if (peek() && peek().t === "+") { pos++; return parseUnary(); }
    return parsePrimary();
  }
  function parsePrimary() {
    const tk = peek();
    if (!tk) return null;
    if (tk.t === "num") { pos++; return tk.v; }
    if (tk.t === "(") {
      pos++;
      const v = parseExpr();
      if (!peek() || peek().t !== ")") return null;
      pos++;
      return v;
    }
    return null;
  }
  const v = parseExpr();
  if (pos !== tokens.length || v === null || !isFinite(v)) return null;
  return v;
}

// Parse "3x^2 + 2x - 5" (also "3*x^2", "x") into [[coef, exp], ...]
function parsePoly(src) {
  let s = src.replace(/\s+/g, "").replace(/\*/g, "");
  if (!/^[0-9x+\-^./()]+$/.test(s) || /[()]/.test(s)) return null;
  if (!s.includes("x")) return null;
  s = s.replace(/-/g, "+-");
  const terms = s.split("+").filter((t) => t !== "");
  const out = [];
  for (const term of terms) {
    if (!term.includes("x")) {
      const c = parseFloat(term);
      if (isNaN(c)) return null;
      out.push([c, 0]);
      continue;
    }
    const parts = term.split("x");
    if (parts.length > 2) return null;
    let coef = parts[0];
    coef = coef === "" || coef === "+" ? 1 : coef === "-" ? -1 : parseFloat(coef);
    if (isNaN(coef)) return null;
    let exp = 1;
    if (parts[1]) {
      if (!parts[1].startsWith("^")) return null;
      exp = parseFloat(parts[1].slice(1));
      if (isNaN(exp)) return null;
    }
    out.push([coef, exp]);
  }
  return out.length ? out : null;
}

function fmtNum(n) {
  if (Math.abs(n - Math.round(n)) < 1e-9) return String(Math.round(n));
  for (let d = 2; d <= 12; d++) {
    const num = Math.round(n * d);
    if (Math.abs(num / d - n) < 1e-9) {
      if (num === 1) return `1/${d}`;
      if (num === -1) return `-1/${d}`;
      return `${num}/${d}`;
    }
  }
  return String(Math.round(n * 1000) / 1000);
}

function fmtPoly(terms) {
  const parts = [];
  for (const [c, e] of terms) {
    if (Math.abs(c) < 1e-12) continue;
    let s = "";
    if (e === 0) s = fmtNum(c);
    else {
      const ac = Math.abs(c);
      const base = "x" + (e === 1 ? "" : "^" + fmtNum(e));
      if (ac === 1) s = base;
      else if (Number.isInteger(1 / ac) && 1 / ac <= 12) s = `${base}/${1 / ac}`;
      else s = fmtNum(ac) + base;
    }
    parts.push([c < 0 ? "-" : "+", s]);
  }
  if (!parts.length) return "0";
  return parts.map(([sign, s], i) => (i === 0 ? (sign === "-" ? "-" + s : s) : ` ${sign} ${s}`)).join("");
}

function integratePoly(terms) {
  return terms.map(([c, e]) => [c / (e + 1), e + 1]);
}

function diffPoly(terms) {
  return terms.map(([c, e]) => [c * e, e - 1]).filter(([, e]) => e >= 0);
}

// Returns answer string or null if not a math query we can solve.
export function localMath(query) {
  const q = query.trim();
  const lower = q.toLowerCase();

  let m = lower.match(/^(integrate|integral of|antiderivative of)\s+(.+)$/);
  if (m) {
    const terms = parsePoly(m[2]);
    if (terms) return `integral ${m[2].trim()} dx = ${fmtPoly(integratePoly(terms))} + constant`;
    return null;
  }
  m = lower.match(/^(differentiate|derivative of|derive)\s+(.+)$/);
  if (m) {
    const terms = parsePoly(m[2]);
    if (terms) return `d/dx ${m[2].trim()} = ${fmtPoly(diffPoly(terms))}`;
    return null;
  }
  // bare arithmetic: must be only numbers/operators/parens and contain an operator
  if (/^[0-9+\-*/^().\s]+$/.test(q) && /[+\-*/^]/.test(q) && !/[a-zA-Z]/.test(q)) {
    const v = safeEval(q);
    if (v !== null) return `${q.trim()} = ${fmtNum(v)}`;
  }
  return null;
}
