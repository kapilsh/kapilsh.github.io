# frozen_string_literal: true

source "https://rubygems.org"

# Pinned exactly. _includes/{sidebar,topbar}.html are copies of this version's
# files with small edits, so a floating version silently desyncs the overrides
# from the theme's CSS/JS. Gemfile.lock is gitignored, so CI re-resolves on every
# build and would otherwise drift on its own. Bump this and re-diff the overrides
# together, never separately. Needs Ruby >= 3.1.
gem "jekyll-theme-chirpy", "7.6.0"

group :test do
  gem "html-proofer", "~> 4.4"
end
