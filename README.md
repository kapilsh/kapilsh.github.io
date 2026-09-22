# kapilsh.github.io

Source for **[www.kapilsharma.dev](https://www.kapilsharma.dev)** — my website and technical blog,
mostly GPUs, PyTorch internals, and whatever I'm reading about kernels that week.

Built with [Jekyll](https://jekyllrb.com/), deployed to GitHub Pages. The theme
comes from a gem named in the `Gemfile` rather than being vendored into the repo.

The theme gem is **pinned to an exact version** there. `Gemfile.lock` is
gitignored, so CI re-resolves dependencies on every build; with a floating
constraint the gem drifted while the `_includes/` overrides stayed on an old
copy, which is how a dead mode-toggle button shipped as an empty circle. Bump the
pin and re-diff the overrides in the same commit, never separately.

## Run it locally

```bash
bin/jekyll serve --livereload   # http://127.0.0.1:4000
bin/jekyll serve --drafts       # include _drafts/
bin/jekyll build
```

`bin/jekyll` exists because plain `jekyll serve` does not work here. The theme is
pinned to an exact version and needs **Ruby >= 3.1**, while the system Ruby is
3.0.2 (end of life since April 2024), so the pinned gem cannot even install
against it. The wrapper runs Jekyll under the Ruby named in `.ruby-version`,
built once with:

```bash
git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build
~/.rbenv/plugins/ruby-build/bin/ruby-build 3.3.12 ~/.rbenv/versions/3.3.12
```

It installs gems into `vendor/bundle` and clears the globally exported
`GEM_HOME`, so nothing it does reaches `~/gems` or the system Ruby. That is
deliberate: wiring rbenv into the shell instead would put gems built against
Ruby 3.0 on a 3.3 load path. Local now builds with the same Jekyll and theme
versions CI does, which is the point — this site previously previewed with
theme 7.0.1 locally while deploying 7.6.x, and the two had drifted far enough
apart that the mode-toggle button rendered as an empty circle in production and
looked fine locally.

Vendored theme JS/CSS lives in the `assets/lib` submodule, so on a fresh clone:

```bash
git submodule update --init
```

Before pushing, it's worth running the same link check CI runs:

```bash
bundle exec jekyll b
bundle exec htmlproofer _site --disable-external \
  --ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"
```

## Layout

| Path | What's in it |
| --- | --- |
| `_posts/` | Published posts, `YYYY-MM-DD-slug.md`. |
| `_drafts/` | Work in progress; no date prefix needed, only rendered with `--drafts`. |
| `_tabs/` | Sidebar tabs. Beyond the theme's About/Archives/Categories/Tags, this holds the self-contained visualizers (`mxfp4.md`, `rope.md`). |
| `_includes/` | Overrides of theme includes — see below. |
| `_layouts/wide-page.html` | Full-width layout used by the visualizer tabs: hides the right-hand TOC panel and lets content span the grid. |
| `_plugins/posts-lastmod-hook.rb` | Sets `last_modified_at` on a post from its git history, when it has more than one commit. |
| `_data/` | `authors.yml`, `contact.yml` (sidebar social icons), `share.yml`. |
| `assets/` | Post images and other static files. `assets/lib/` is the theme-assets submodule. |
| `CNAME` | Custom domain: `www.kapilsharma.dev`. |

## Writing a post

Drop a file in `_posts/` named `YYYY-MM-DD-slug.md` with the usual front matter:

```yaml
---
title: A Title
date: 2026-01-01 10:00:00 -0500
categories: [GPU, CUDA]
tags: [cuda, triton]
image: /assets/cover.png
math: true       # only if you use $$ ... $$
---
```

Images go in `assets/` and are referenced as `/assets/name.png`.

## Theme overrides

`_includes/` shadows files of the same name in the gem. Each override carries a header comment
saying what was changed and why — the rule is to keep the diff against the gem as small as
possible and **re-diff after any theme upgrade**:

```bash
theme=$(sed -n 's/^gem "\([^"]*\)".*/\1/p' Gemfile | head -1)
diff "$(bundle info "$theme" --path)/_includes/sidebar.html" _includes/sidebar.html
```

- **`sidebar.html`** — defines the `#ks-mark` brand SVG (used here and by `topbar.html`), swaps it
  in for the home icon, and adds the standalone-app links below the tab list.
- **`topbar.html`** — puts the brand mark before the breadcrumb and the mobile title.
- **`footer.html`** — drops the trailing "powered by" paragraph, leaving only the
  copyright line.

## Standalone apps

The apps linked at the bottom of the sidebar each live in their own repo and publish their own
GitHub Pages project site, so they are **not** part of this build:

| App | Repo | URL |
| --- | --- | --- |
| Shardlock | [`kapilsh/shardlock`](https://github.com/kapilsh/shardlock) | <https://www.kapilsharma.dev/shardlock/> |
| Perfessor | [`kapilsh/perfessor`](https://github.com/kapilsh/perfessor) | <https://www.kapilsharma.dev/perfessor/> |
| CUDA Quiz | [`kapilsh/cuda-quiz`](https://github.com/kapilsh/cuda-quiz) | <https://www.kapilsharma.dev/cuda-quiz/> |
| Nano Shards | [`kapilsh/nano-shards`](https://github.com/kapilsh/nano-shards) | <https://www.kapilsharma.dev/nano-shards/> |

To add one, append a `nav-item` in `_includes/sidebar.html`. Use the **absolute** URL on the
canonical domain, not a root-relative path: htmlproofer resolves internal links against `_site`
and would flag `/foo/` as broken, while external links are skipped under `--disable-external`.

## Deploy

`.github/workflows/pages-deploy.yml` runs on every push to `master`: build → htmlproofer →
deploy to GitHub Pages. Pushes that only touch `README.md`, `LICENSE`, or `.gitignore` are
skipped, and the workflow can also be run manually from the Actions tab.

## License

[MIT](LICENSE) for the code. Post content is © Kapil Sharma.
