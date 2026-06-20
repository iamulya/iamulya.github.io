# iamulya.one — Personal Blog

Built with [Jekyll](https://jekyllrb.com/) and the [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme. Deployed via GitHub Pages.

## Local Development

### Prerequisites

```bash
bundle install
```

### Running the dev server

**Standard (full rebuild every time, ~40s):**
```bash
bundle exec jekyll serve
```

**Recommended for active writing (~1–2s rebuilds after the first):**
```bash
bundle exec jekyll serve --incremental --livereload --config _config.yml,_config_dev.yml
```

The `--incremental` flag means only the file you edited is rebuilt. The first build on startup still takes ~40s (Rouge syntax highlighting all code blocks across every post), but every subsequent rebuild when you save a file is near-instant.

> **Note:** `--incremental` does not detect changes to `_config.yml`, `_includes/`, or `_layouts/`. If you edit those, stop the server and restart.

### Why is the full build slow?

~40s is expected for this site. The bottleneck is [Rouge](https://github.com/rouge-ruby/rouge) syntax highlighting — it parses every fenced code block in every post on every full build. With 20+ long technical articles each containing many code blocks, this dominates build time. Liquid template rendering is only ~1.3s of the total.

| What | Time | Notes |
|---|---|---|
| Rouge syntax highlighting | ~27s | All code blocks across all posts |
| Liquid template rendering | ~1.3s | Jekyll `--profile` output |
| Assets, Sass, file copy | ~2–3s | One-time per full build |

### Dev config (`_config_dev.yml`)

The `_config_dev.yml` file disables a few expensive production features for faster local iteration:
- Sass compression disabled (`style: expanded`)
- Google Analytics injection skipped
- PWA cache generation disabled (~1s saving)

### Building for production

GitHub Actions handles production builds automatically on push. To validate locally:

```bash
bundle exec jekyll build JEKYLL_ENV=production
```

### Checking for broken links

The GitHub Actions workflow runs `html-proofer` on each build. To run it locally:

```bash
bundle exec jekyll build && bundle exec htmlproofer ./_site \
  --disable-external \
  --ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"
```

## Content Structure

```
_posts/          Blog posts (YYYY-MM-DD-slug.md)
_drafts/         Unpublished drafts
assets/img/      Post images
_config.yml      Main site config
_config_dev.yml  Development overrides (faster local builds)
```

### Series

| Series | Tag | Articles |
|---|---|---|
| Generative AI in Depth | `Generative AI in Depth` | 15 parts (Parts 1–15) |
| vLLM Deep Dive | `vLLM Deep Dive Series` | 3 parts |

### Post front matter conventions

- `categories: [Generative AI in Depth]` — use the exact string; Jekyll archives slugifies it to `/categories/generative-ai-in-depth/`
- `tags: [...]` — all lowercase except proper acronyms (LLM, GPU) and series names
- `image.path` — always `/assets/img/<filename>` (not URL-encoded, lowercase with hyphens)
- `date` — must not be in the future, or Jekyll skips the post at build time

## Deployment

Pushes to `main` trigger the GitHub Actions workflow which builds and deploys to GitHub Pages at [iamulya.one](https://iamulya.one).