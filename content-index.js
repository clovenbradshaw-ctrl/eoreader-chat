#!/usr/bin/env node
// content-index.js — Structural codebase index for intelligent high-level traversal.
//
// Builds a queryable in-memory index across all workspace repos capturing:
//   1. REPO LEVEL — each repo, purpose, package deps
//   2. PACKAGE LEVEL — package.json structure, entry points
//   3. MODULE LEVEL — files, imports, exports, definitions
//   4. ENTITY LEVEL — eoreader5 conceptual entities → implementation files
//   5. CROSS-REF LEVEL — import/export graph between modules
//
// Query methods: find, lookup, related, structure, entities, search, graph
//
// Usage:
//   import { ContentIndex } from "./content-index.js";
//   const idx = new ContentIndex();
//   await idx.scan(["/path/to/eoreader5", "/path/to/eoPriors", ...]);
//   const results = idx.find("search");
//   const info = idx.lookup("packages/engine/search/index.js");
//   const tree = idx.structure("packages/engine/emergence");
//   const rel = idx.related("packages/engine/emergence/store/index.js");

import fs from "fs";
import path from "path";

// ── Patterns ────────────────────────────────────────────────────────────────

const IMPORT_PAT = /import\s+(?:[\w*{}\s,]+\s+from\s+)?["']([^"']+)["']/g;
const EXPORT_PAT = /export\s+(?:(?:default\s+)?(?:function\s+(\w+)|class\s+(\w+)|const\s+(\w+)|let\s+(\w+)|var\s+(\w+)))/g;
const EXPORT_NAMED_PAT = /export\s+\{\s*([^}]+)\s*\}/g;
const RE_EXPORT_PAT = /export\s+(?:\{[^}]*\})?\s*from\s+["']([^"']+)["']/g;
const FUNC_PAT = /(?:async\s+)?function\s+(\w+)\s*\(/g;
const CLASS_PAT = /class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{/g;
const CONST_PAT = /(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|async\s*\(|function\s*\()/g;
const ASSIGN_PAT = /(?:this\.|module\.exports\s*=\s*)\s*(\w+)\s*[:=]/g;
const HEADER_PAT = /\/\/\s*(.+?)(?:\n\/\/\s*(.*?))*(?:\n\n|\nimport|\nexport)/s;

const IGNORE_DIRS = new Set(["node_modules", ".git", "dist", "build", "__pycache__", ".opencode", "_archive", "archive", "coverage", ".nyc_output", "target", ".next", ".venv", ".mypy_cache", ".pytest_cache"]);
const SKIP_EXTS = new Set([".json", ".lock", ".map", ".png", ".jpg", ".gif", ".ico", ".svg", ".woff", ".woff2", ".mp3", ".mp4", ".wasm", ".bin", ".exe", ".dll", ".so", ".dylib", ".ttf", ".otf", ".eot"]);

const ENTITY_NAMES = new Map([
  ["cube", "cube classifier"],
  ["presence", "referent presence"],
  ["store", "associative memory"],
  ["entity-fold", "entity fold"],
  ["multi-altitude-fold", "multi-altitude fold"],
  ["discourse", "discourse"],
  ["spine", "significance spine"],
  ["reaction", "reaction channel"],
  ["search", "search"],
  ["retrieval", "retrieval"],
  ["structural-query", "structural query"],
  ["fold", "fold compression"],
  ["salience", "born salience"],
  ["surprise", "surprise measure"],
  ["nulls", "null derivation"],
  ["calculus", "calculus induction"],
  ["trajectory", "trajectory red shift"],
  ["lens-assertion", "lens assertion"],
  ["reader-priors", "reader priors"],
  ["genesis", "task genesis"],
  ["chapters", "chapter detection"],
  ["boundaries", "boundary detection"],
  ["parameters", "parameter profiles"],
  ["entity-kinds", "entity kinds induction"],
  ["projection", "projection"],
  ["motif", "motif detection"],
  ["quantum", "quantum engine"],
  ["replay", "replay ledger"],
  ["cube/index.js", "cube classifier"],
  ["presence.js", "referent presence organ"],
  ["store/index.js", "associative memory organ"],
  ["entity-fold.js", "entity fold organ"],
  ["multi-altitude-fold.js", "multi-altitude fold organ"],
  ["discourse/index.js", "discourse organ"],
  ["spine.js", "significance spine organ"],
  ["reaction/index.js", "reaction channel organ"],
  ["search/index.js", "engine search organ"],
  ["retrieval/index.js", "signal retrieval organ"],
  ["structural-query/index.js", "structural query organ"],
  ["observation-index.js", "observation index"],
  ["referents/index.js", "referent resolution"],
  ["ledger/index.js", "semantic ledger"],
  ["prediction", "predictive competency"],
  ["competency", "competency ledger"],
  ["perceiver/text/presence.js", "referent presence organ"],
  ["perceiver/text/text-signal.js", "text signal extraction"],
  ["perceiver/audio", "audio perception"],
  ["perceiver/video", "video perception"],
  ["perceiver/dispatch.js", "perception dispatch"],
  ["emergence/summary/index.js", "summary engine"],
  ["emergence/evaluate/index.js", "evaluate"],
  ["emergence/store/index.js", "associative memory organ"],
  ["emergence/fold/index.js", "fold compression"],
]);

const REPO_DESCRIPTIONS = {
  "eoreader5": "Current semantic engine — all organs (cube, presence, fold, store, discourse, spine, reaction) and the multi-altitude entity summary oracle",
  "eoPriors": "Priors: corpus prior cube, per-text coref alias/narrator knowledge, injected as witness-tier priors",
  "eoreader4.2": "Legacy engine — src/weave/write/ has the unported phraser→talker prose pipeline",
  "eoreader-chat": "Chat interface and proxy — connects users/LLMs to the eoreader5 engine via MCP tools and a BoundedStore memory layer",
  "eoreader-proxy": "Proxy server (if separate from chat)",
  "eoreaderapp": "Browser application — in-browser eoreader5 engine, html UI, test harness",
  "eoreader-mcp": "MCP server — JSON-RPC stdio server exposing eoreader5 tools (ingest, scout, fold, think, speak, etc.) to opencode and other MCP clients",
};

// ── Helpers ─────────────────────────────────────────────────────────────────

function extractModuleHeader(text) {
  const firstLines = text.split("\n").slice(0, 20);
  const lines = [];
  for (const l of firstLines) {
    const trimmed = l.replace(/^\/\/\s*/, "").trim();
    if (trimmed && !trimmed.startsWith("!")) lines.push(trimmed);
    if (l.startsWith("import") || l.startsWith("export") || (lines.length > 0 && l.trim() === "" && lines.length > 3)) break;
  }
  return lines.slice(0, 6).join(" ").replace(/\s+/g, " ").trim();
}

function parseImports(text) {
  const imports = new Set();
  let m;
  IMPORT_PAT.lastIndex = 0;
  while ((m = IMPORT_PAT.exec(text)) !== null) {
    const target = m[1] || m[2] || m[3];
    if (target) imports.add(target);
  }
  return [...imports];
}

function parseExports(text) {
  const exports = [];
  let m;
  EXPORT_PAT.lastIndex = 0;
  while ((m = EXPORT_PAT.exec(text)) !== null) {
    const name = m[1] || m[2] || m[3] || m[4] || m[5];
    const type = m[1] ? "function" : m[2] ? "class" : "const";
    if (name) exports.push({ name, type, line: lineAt(text, m.index) });
  }
  EXPORT_NAMED_PAT.lastIndex = 0;
  while ((m = EXPORT_NAMED_PAT.exec(text)) !== null) {
    for (const name of m[1].split(",").map(s => s.trim().split(/\s+as\s+/)[0].trim()).filter(Boolean)) {
      exports.push({ name, type: "named", line: lineAt(text, m.index) });
    }
  }
  return exports;
}

function parseDefinitions(text) {
  const defs = [];
  let m;
  FUNC_PAT.lastIndex = 0;
  while ((m = FUNC_PAT.exec(text)) !== null) defs.push({ name: m[1], type: "function", line: lineAt(text, m.index) });
  CLASS_PAT.lastIndex = 0;
  while ((m = CLASS_PAT.exec(text)) !== null) defs.push({ name: m[1], type: "class", line: lineAt(text, m.index) });
  CONST_PAT.lastIndex = 0;
  while ((m = CONST_PAT.exec(text)) !== null) defs.push({ name: m[1], type: "function", line: lineAt(text, m.index) });
  return defs;
}

function lineAt(text, index) {
  return text.slice(0, index).split("\n").length;
}

function extractDeps(pkg) {
  return { ...(pkg.dependencies || {}), ...(pkg.devDependencies || {}) };
}

function resolveImportPath(imp, moduleDir, pkg, repoRoot) {
  if (imp.startsWith(".")) {
    const resolved = path.resolve(moduleDir, imp);
    // Normalize to repo-relative path
    try {
      const rel = path.relative(repoRoot, resolved);
      if (!rel.startsWith("..")) return rel.replace(/\\/g, "/");
    } catch {}
    return resolved.replace(/\\/g, "/");
  }
  if (pkg?.dependencies?.[imp] || pkg?.devDependencies?.[imp]) return `pkg:${imp}`;
  if (imp.startsWith("@")) {
    const scoped = imp.split("/").slice(0, 2).join("/");
    if (pkg?.dependencies?.[scoped] || pkg?.devDependencies?.[scoped]) return `pkg:${scoped}`;
  }
  return `ext:${imp}`;
}

function classifyEntities(relPath, text) {
  const entities = [];
  const parts = relPath.split("/");
  const base = parts[parts.length - 1];

  // Check path-ending matches first: exact filename or full path suffix
  for (const [key, name] of ENTITY_NAMES) {
    if (key.includes("/") && relPath.endsWith(key)) {
      entities.push(name);
    } else if (base === key) {
      entities.push(name);
    }
  }

  // Then check parts-level directory matches (whole path segment, not substring)
  for (const [key, name] of ENTITY_NAMES) {
    if (key.includes("/")) continue;
    if (parts.length > 1 && parts.some(p => p === key)) {
      if (!entities.includes(name)) entities.push(name);
    }
  }

  return [...new Set(entities)];
}

// ── ContentIndex ────────────────────────────────────────────────────────────

export class ContentIndex {
  constructor() {
    this.repos = new Map();
    this.entities = new Map();
    this.definitions = new Map();
    this.crossRefs = new Map();
    this.names = new Map();
    this.allText = new Map();
    this.built = false;
    this.scanTime = 0;
    this.totalFiles = 0;
  }

  async scan(rootPaths) {
    const start = Date.now();
    this.totalFiles = 0;

    for (const root of rootPaths) {
      const repoName = path.basename(root);
      const repoInfo = {
        name: repoName,
        path: root,
        description: REPO_DESCRIPTIONS[repoName] || "",
        packages: new Map(),
        totalFiles: 0,
      };

      // Read repo-level metadata
      const readmePath = path.join(root, "README.md");
      if (fs.existsSync(readmePath)) {
        try {
          repoInfo.description = fs.readFileSync(readmePath, "utf8").split("\n").slice(0, 5).join(" ").replace(/#/g, "").trim();
        } catch {}
      }

      // Find and scan all packages
      const pkgPaths = await this._findPackageDirs(root);
      if (pkgPaths.length === 0) {
        // Single-package repo (root IS the package)
        pkgPaths.push(root);
      }

      for (const pkgDir of pkgPaths) {
        const pkgJsonPath = path.join(pkgDir, "package.json");
        let pkg = { name: path.basename(pkgDir), main: "index.js" };
        try {
          pkg = { ...pkg, ...JSON.parse(fs.readFileSync(pkgJsonPath, "utf8")) };
        } catch {}

        const pkgName = pkg.name || path.basename(pkgDir);
        const pkgRel = path.relative(root, pkgDir);
        const pkgInfo = {
          name: pkgName,
          path: pkgDir,
          main: pkg.main || "index.js",
          deps: extractDeps(pkg),
          modules: new Map(),
        };

        // Scan modules in the package
        const moduleFiles = await this._walkDir(pkgDir, pkgDir);

        for (const modPath of moduleFiles) {
          const relPath = path.relative(pkgDir, modPath);
          const repoRel = path.relative(root, modPath);
          try {
            const text = fs.readFileSync(modPath, "utf8");
            if (text.length < 20) continue;
            this.totalFiles++;

            // Store full text for content search
            this.allText.set(`${repoName}/${repoRel}`, text);

            const imports = parseImports(text);
            const exports = parseExports(text);
            const defs = parseDefinitions(text);
            const header = extractModuleHeader(text);
            const entities = classifyEntities(relPath, text);

            const modInfo = {
              path: modPath,
              relPath,
              repoRel,
              repoName,
              pkgName,
              size: text.length,
              lines: text.split("\n").length,
              header,
              imports,
              exports,
              definitions: defs,
              entities,
            };
            pkgInfo.modules.set(relPath, modInfo);

            // Index definitions globally
            for (const def of defs) {
              if (!this.definitions.has(def.name)) this.definitions.set(def.name, []);
              this.definitions.get(def.name).push({
                repo: repoName,
                pkg: pkgName,
                path: repoRel,
                line: def.line,
                type: def.type,
              });
            }

            // Index named exports globally
            for (const exp of exports) {
              if (!this.names.has(exp.name)) this.names.set(exp.name, []);
              this.names.get(exp.name).push({
                repo: repoName,
                pkg: pkgName,
                path: repoRel,
                line: exp.line,
                type: exp.type,
                kind: "export",
              });
            }

            // Index entities
            for (const entity of entities) {
              if (!this.entities.has(entity)) this.entities.set(entity, { name: entity, files: [], description: "" });
              this.entities.get(entity).files.push({
                repo: repoName,
                pkg: pkgName,
                path: repoRel,
                role: "implements",
              });
            }

            // Cross-references: resolve imports to paths
            const moduleDir = path.dirname(modPath);
            for (const imp of imports) {
              const resolved = resolveImportPath(imp, moduleDir, pkg, root);
              if (!this.crossRefs.has(repoRel)) this.crossRefs.set(repoRel, { imports: [], importedBy: [] });
              this.crossRefs.get(repoRel).imports.push({ spec: imp, resolved });
            }
          } catch {}
        }

        repoInfo.packages.set(pkgName, pkgInfo);
        repoInfo.totalFiles += pkgInfo.modules.size;
      }

      this.repos.set(repoName, repoInfo);
    }

    // Build reverse cross-refs (who imports who)
    // Collect every known module path
    const allModPaths = new Set(this.crossRefs.keys());
    for (const repo of this.repos.values()) {
      for (const pkg of repo.packages.values()) {
        for (const relPath of pkg.modules.keys()) {
          allModPaths.add(relPath);
        }
      }
    }

    for (const [repoRel, refs] of this.crossRefs) {
      for (const imp of refs.imports) {
        const impPath = imp.resolved.replace(/\\/g, "/");
        if (impPath.startsWith("ext:") || impPath.startsWith("pkg:")) continue;

        const impNoExt = impPath.replace(/\.(js|mjs|ts)$/, "");

        for (const otherPath of allModPaths) {
          if (otherPath === repoRel) continue;
          const otherNoExt = otherPath.replace(/\.(js|mjs|ts)$/, "");
          if (otherNoExt === impNoExt || otherNoExt.endsWith("/" + impNoExt)) {
            let entry = this.crossRefs.get(otherPath);
            if (!entry) {
              entry = { imports: [], importedBy: [] };
              this.crossRefs.set(otherPath, entry);
            }
            if (!entry.importedBy.some(e => e.by === repoRel)) {
              entry.importedBy.push({ by: repoRel });
            }
          }
        }
      }
    }

    // Derive entity descriptions from AGENTS.md content
    for (const repo of this.repos.values()) {
      const agentsPath = path.join(repo.path, "AGENTS.md");
      if (fs.existsSync(agentsPath)) {
        try {
          const agents = fs.readFileSync(agentsPath, "utf8");
          // Extract the organ table
          const tableMatch = agents.match(/\| organ \| file \| status \|[\s\S]*?(?=\n##|\n$)/);
          if (tableMatch) {
            const lines = tableMatch[0].split("\n").filter(l => l.startsWith("|") && l.split("|").length >= 4);
            for (const line of lines.slice(2)) { // skip header + separator
              const parts = line.split("|").map(s => s.trim());
              if (parts.length >= 4) {
                const organName = parts[1];
                const filePath = parts[2];
                const status = parts[3];
                const entName = organName.replace(/`/g, "").trim();
                if (this.entities.has(entName)) {
                  this.entities.get(entName).description = status;
                }
                // Also match by file path
                for (const [eName, eInfo] of this.entities) {
                  if (eInfo.files.some(f => f.path.includes(filePath.replace(/`/g, "")))) {
                    eInfo.description = status;
                  }
                }
              }
            }
          }
        } catch {}
      }
    }

    this.built = true;
    this.scanTime = Date.now() - start;
  }

  // ── Find packages ──

  async _findPackageDirs(dir) {
    const results = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
      if (IGNORE_DIRS.has(e.name)) continue;
      if (e.isDirectory()) {
        const pkgPath = path.join(dir, e.name);
        const pkgJson = path.join(pkgPath, "package.json");
        if (fs.existsSync(pkgJson)) {
          results.push(pkgPath);
        }
      }
    }
    return results;
  }

  async _walkDir(dir, baseDir, results = []) {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
    catch { return results; }

    for (const e of entries) {
      if (e.name.startsWith(".") || IGNORE_DIRS.has(e.name)) continue;
      const fullPath = path.join(dir, e.name);
      try {
        if (e.isDirectory()) {
          await this._walkDir(fullPath, baseDir, results);
        } else if (e.isFile()) {
          const ext = path.extname(e.name);
          if (SKIP_EXTS.has(ext) || e.name.includes(".test.") || e.name.includes(".spec.")) continue;
          results.push(fullPath);
        }
      } catch {}
    }
    return results;
  }

  // ── Query: find ───────────────────────────────────────────────────────────

  find(term, options = {}) {
    if (!this.built) return { error: "index not built" };
    const { limit = 20, contextLines = 0 } = options;
    const lower = term.toLowerCase();
    const results = [];

    // 1. Match definitions (functions, classes, consts)
    for (const [name, locs] of this.definitions) {
      if (name.toLowerCase().includes(lower)) {
        for (const loc of locs) {
          results.push({ type: "definition", name, ...loc, excerpt: this._getLine(loc.path, loc.line) });
        }
      }
    }

    // 2. Match exports
    for (const [name, locs] of this.names) {
      if (name.toLowerCase().includes(lower) && !results.some(r => r.type === "export" && r.name === name && r.path === locs[0]?.path)) {
        for (const loc of locs) {
          results.push({ type: "export", name, ...loc, excerpt: this._getLine(loc.path, loc.line) });
        }
      }
    }

    // 3. Match module headers
    for (const repo of this.repos.values()) {
      for (const pkg of repo.packages.values()) {
        for (const [relPath, mod] of pkg.modules) {
          if (mod.header.toLowerCase().includes(lower) || relPath.toLowerCase().includes(lower)) {
            results.push({ type: "module", name: relPath, path: mod.repoRel, repo: repo.name, pkg: pkg.name, header: mod.header, entities: mod.entities, size: mod.size, lines: mod.lines });
          }
        }
      }
    }

    // 4. Match entities
    for (const [entName, entInfo] of this.entities) {
      if (entName.toLowerCase().includes(lower)) {
        results.push({ type: "entity", name: entName, description: entInfo.description, files: entInfo.files });
      }
    }

    // 5. Content match in source text
    for (const [storePath, text] of this.allText) {
      const lines = text.split("\n");
      const slashIdx = storePath.indexOf("/");
      const repoPart = slashIdx > 0 ? storePath.slice(0, slashIdx) : "?";
      const pathPart = slashIdx > 0 ? storePath.slice(slashIdx + 1) : storePath;
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].toLowerCase().includes(lower)) {
          const start = Math.max(0, i - contextLines);
          const end = Math.min(lines.length, i + contextLines + 1);
          results.push({ type: "content", name: pathPart, path: storePath, repo: repoPart, line: i + 1, context: lines.slice(start, end).join("\n") });
          if (results.filter(r => r.type === "content").length >= limit) break;
        }
      }
      if (results.filter(r => r.type === "content").length >= limit) break;
    }

    // Deduplicate and sort by relevance
    const scored = results.map(r => {
      let score = 0;
      if (r.type === "definition") score = 100;
      else if (r.type === "export") score = 80;
      else if (r.type === "entity") score = 90;
      else if (r.type === "module") score = 70;
      else if (r.type === "content") score = 30;
      // Exact matches rank higher
      if (r.name && r.name.toLowerCase() === lower) score += 50;
      return { ...r, score };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  // ── Query: lookup ─────────────────────────────────────────────────────────

  lookup(repoRelPath) {
    if (!this.built) return { error: "index not built" };
    const norm = repoRelPath.replace(/\\/g, "/");

    for (const repo of this.repos.values()) {
      for (const pkg of repo.packages.values()) {
        for (const [relPath, mod] of pkg.modules) {
          if (mod.repoRel === norm || relPath === norm || mod.repoRel.endsWith(norm)) {
            const refs = this.crossRefs.get(mod.repoRel) || { imports: [], importedBy: [] };
            const entList = mod.entities.map(e => this.entities.get(e)).filter(Boolean);
            return {
              ...mod,
              imports: refs.imports,
              importedBy: refs.importedBy,
              entityDetails: entList,
            };
          }
        }
      }
    }
    return null;
  }

  // ── Query: related ──────────────────────────────────────────────────────────

  related(repoRelPath) {
    if (!this.built) return { error: "index not built" };
    const mod = this.lookup(repoRelPath);
    if (!mod) return { error: "module not found" };

    const refs = this.crossRefs.get(mod.repoRel) || { imports: [], importedBy: [] };

    // Resolve imported modules
    const importedModules = [];
    for (const imp of refs.imports) {
      const resolved = this.lookup(imp.resolved);
      if (resolved) importedModules.push(resolved);
    }

    // Modules that import this one
    const dependentModules = [];
    for (const ref of refs.importedBy) {
      const resolved = this.lookup(ref.by);
      if (resolved) dependentModules.push(resolved);
    }

    return {
      module: mod,
      imports: importedModules,
      importedBy: dependentModules,
      entities: mod.entityDetails || [],
      crossRefs: refs,
    };
  }

  // ── Query: structure (tree) ────────────────────────────────────────────────

  structure(repoRelPrefix) {
    if (!this.built) return { error: "index not built" };
    const prefix = repoRelPrefix ? repoRelPrefix.replace(/\\/g, "/") : "";
    const tree = { name: prefix || "(root)", type: "dir", children: [] };
    const seen = new Set();

    for (const repo of this.repos.values()) {
      for (const pkg of repo.packages.values()) {
        for (const [relPath, mod] of pkg.modules) {
          if (!mod.repoRel.startsWith(prefix) && prefix) continue;
          if (seen.has(mod.repoRel)) continue;
          seen.add(mod.repoRel);

          const parts = mod.repoRel.replace(prefix, "").replace(/^\//, "").split("/");
          let current = tree;
          for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            const isFile = i === parts.length - 1 && part.includes(".");
            let child = current.children.find(c => c.name === part);
            if (!child) {
              child = isFile
                ? { name: part, type: "file", path: mod.repoRel, header: mod.header, entities: mod.entities, defs: mod.definitions.length, exports: mod.exports.length, lines: mod.lines }
                : { name: part, type: "dir", children: [] };
              current.children.push(child);
            }
            current = child;
          }
        }
      }
    }

    tree.children.sort((a, b) => {
      if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    return tree;
  }

  // ── Query: entities ──────────────────────────────────────────────────────

  entityIndex() {
    if (!this.built) return { error: "index not built" };
    const result = {};
    for (const [name, info] of this.entities) {
      result[name] = {
        name,
        description: info.description,
        files: info.files,
        fileCount: info.files.length,
      };
    }
    return result;
  }

  // ── Query: search (full-text + structural) ───────────────────────────────

  search(query, options = {}) {
    if (!this.built) return { error: "index not built" };
    const { limit = 20, repo: repoFilter } = options;
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);

    if (terms.length === 0) return { error: "empty query" };

    const scored = [];

    // Score each module by term density
    for (const repo of this.repos.values()) {
      if (repoFilter && repo.name !== repoFilter) continue;
      for (const pkg of repo.packages.values()) {
        for (const [relPath, mod] of pkg.modules) {
          const text = this.allText.get(mod.repoRel) || "";
          const lower = text.toLowerCase();
          let termHits = 0;
          let exactPhrase = 0;

          // Check each term
          for (const t of terms) {
            let idx = -1;
            let count = 0;
            while ((idx = lower.indexOf(t, idx + 1)) !== -1) count++;
            termHits += count;
          }

          // Check exact phrase match
          if (lower.includes(query.toLowerCase())) exactPhrase = 5;

          // Check header match
          const headerScore = mod.header.toLowerCase().includes(query.toLowerCase()) ? 20 : 0;

          // Definition/export name matches
          const defScore = mod.definitions.filter(d => d.name.toLowerCase().includes(query.toLowerCase())).length * 10;
          const expScore = mod.exports.filter(e => e.name.toLowerCase().includes(query.toLowerCase())).length * 8;

          if (termHits > 0 || headerScore > 0 || defScore > 0) {
            scored.push({
              path: mod.repoRel,
              repo: repo.name,
              pkg: pkg.name,
              header: mod.header,
              entities: mod.entities,
              score: termHits * 2 + exactPhrase * 10 + headerScore + defScore + expScore,
              termHits,
              lines: mod.lines,
              definitions: mod.definitions.length,
              exports: mod.exports.length,
            });
          }
        }
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  // ── Query: graph (dependency graph for a module) ─────────────────────────

  graph(repoRelPath, options = {}) {
    const { depth = 1 } = options;
    if (!this.built) return { error: "index not built" };

    const mod = this.lookup(repoRelPath);
    if (!mod) return { error: "module not found" };

    const visited = new Set();
    function traverse(currentPath, currentDepth) {
      if (currentDepth > depth || visited.has(currentPath)) return null;
      visited.add(currentPath);

      const info = this.lookup(currentPath);
      if (!info) return { path: currentPath, error: "not in index" };

      const refs = this.crossRefs.get(info.repoRel) || { imports: [], importedBy: [] };
      return {
        path: info.repoRel,
        header: info.header,
        entities: info.entities,
        imports: refs.imports.map(i => traverse.call(this, i.resolved, currentDepth + 1)).filter(Boolean),
        importedBy: refs.importedBy.map(i => traverse.call(this, i.by, currentDepth + 1)).filter(Boolean),
      };
    }

    return traverse.call(this, repoRelPath, 0);
  }

  // ── Query: export API surface ─────────────────────────────────────────────

  apiSurface(repoRelPrefix) {
    if (!this.built) return { error: "index not built" };
    const prefix = repoRelPrefix ? repoRelPrefix.replace(/\\/g, "/") : "";
    const apis = [];

    for (const repo of this.repos.values()) {
      for (const pkg of repo.packages.values()) {
        for (const [relPath, mod] of pkg.modules) {
          if (!mod.repoRel.startsWith(prefix) && prefix) continue;
          if (mod.exports.length > 0 || mod.definitions.length > 0 || mod.entities.length > 0) {
            apis.push({
              path: mod.repoRel,
              repo: repo.name,
              pkg: pkg.name,
              header: mod.header,
              exports: mod.exports,
              definitions: mod.definitions,
              entities: mod.entities,
            });
          }
        }
      }
    }

    return apis;
  }

  // ── Summary ────────────────────────────────────────────────────────────────

  summary() {
    if (!this.built) return { error: "index not built" };
    const repoStats = [];
    for (const [name, repo] of this.repos) {
      let files = 0, lines = 0;
      for (const pkg of repo.packages.values()) {
        for (const mod of pkg.modules.values()) {
          files++;
          lines += mod.lines;
        }
      }
      repoStats.push({ name, description: repo.description, files, lines, packages: repo.packages.size });
    }
    return {
      repos: repoStats,
      totalFiles: this.totalFiles,
      totalEntities: this.entities.size,
      totalDefinitions: this.definitions.size,
      scanTime: this.scanTime,
    };
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  _getLine(repoRel, lineNum) {
    const text = this.allText.get(repoRel);
    if (!text) return "";
    const lines = text.split("\n");
    return lines[lineNum - 1]?.trim()?.slice(0, 120) || "";
  }
}
