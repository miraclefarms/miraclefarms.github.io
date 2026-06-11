require "fileutils"
require "minitest/autorun"
require "tmpdir"
require "yaml"
require "jekyll"

class PostInteractionsTest < Minitest::Test
  ROOT = File.expand_path("..", __dir__)

  def setup
    @site_dir = Dir.mktmpdir("miraclefarms-site-")
    @dest_dir = Dir.mktmpdir("miraclefarms-dest-")

    FileUtils.cp(File.join(ROOT, "_config.yml"), @site_dir)
    FileUtils.cp_r(File.join(ROOT, "_layouts"), @site_dir)
    FileUtils.cp_r(File.join(ROOT, "_includes"), @site_dir) if Dir.exist?(File.join(ROOT, "_includes"))
    FileUtils.cp_r(File.join(ROOT, "assets"), @site_dir)
    FileUtils.mkdir_p(File.join(@site_dir, "_posts"))
    FileUtils.mkdir_p(File.join(@site_dir, "briefs"))
    FileUtils.cp(File.join(ROOT, "briefs", "index.html"), File.join(@site_dir, "briefs"))

    write_post(
      "2026-05-18-interaction-essay.md",
      kind: "essay",
      category: "Essay",
      body: "开篇提出一个问题。\n\n## 一、第一节\n\n这是一段可被读者划线的长文内容。"
    )
    write_post(
      "2026-05-18-interaction-essay.en.md",
      kind: "essay",
      category: "Essay",
      body: "Opening question in English.\n\n## 1. First section\n\nEnglish body paragraph."
    )
    write_post(
      "2026-05-18-interaction-brief.md",
      kind: "brief",
      category: "Brief",
      body: "简报开头。\n\n## 一、今日要点\n\n这是一段简报内容。"
    )
    write_post(
      "2026-05-18-interaction-brief.en.md",
      kind: "brief",
      category: "Brief",
      body: "English brief.\n\n## 1. Point\n\nEnglish content."
    )
    write_post(
      "2026-05-18-interaction-solo.md",
      kind: "essay",
      category: "Essay",
      body: "只有中文，没有英文版。\n\n## 一、唯一一节\n\n正文内容。"
    )
  end

  def teardown
    FileUtils.remove_entry(@site_dir) if @site_dir && Dir.exist?(@site_dir)
    FileUtils.remove_entry(@dest_dir) if @dest_dir && Dir.exist?(@dest_dir)
  end

  def test_long_form_posts_render_lazy_comments_and_reader_highlights
    build_site

    html = rendered_post("interaction-essay")

    assert_includes html, 'id="post-comments"'
    assert_includes html, 'data-giscus-repo="miraclefarms/miraclefarms.github.io"'
    assert_includes html, 'data-giscus-category="General"'
    assert_includes html, "/assets/js/comments-loader.js"
    refute_includes html, "https://giscus.app/client.js"

    assert_includes html, 'data-reader-highlights="true"'
    assert_includes html, "/assets/js/reader-highlights.js"
    assert_includes html, 'id="reader-highlights-list"'
  end

  def test_briefs_skip_comments_but_keep_private_highlights
    build_site

    html = rendered_post("interaction-brief")

    refute_includes html, 'id="post-comments"'
    refute_includes html, "/assets/js/comments-loader.js"
    assert_includes html, 'data-reader-highlights="true"'
    assert_includes html, "/assets/js/reader-highlights.js"
  end

  def test_private_highlights_stay_browser_local
    source = File.read(File.join(ROOT, "assets/js/reader-highlights.js"))

    assert_includes source, "localStorage"
    refute_match(/fetch\s*\(/, source)
    refute_match(/XMLHttpRequest/, source)
    refute_match(/sendBeacon/, source)
  end

  def test_private_highlights_can_delete_individual_saved_marks
    source = File.read(File.join(ROOT, "assets/js/reader-highlights.js"))

    assert_includes source, "data-reader-delete-highlight"
    assert_includes source, 'data-reader-action="delete"'
    assert_includes source, "findHighlightForRange"
    assert_includes source, "deleteHighlight"
    assert_includes source, "删除"
  end

  def test_configured_cloudflare_analytics_loads_in_production
    config = YAML.load_file(File.join(@site_dir, "_config.yml"))
    token = config.fetch("analytics").fetch("cloudflare_token")

    build_site(env: "production")

    html = rendered_post("interaction-essay")

    assert_includes html, "data-site-analytics-loader"
    assert_includes html, "static.cloudflareinsights.com/beacon.min.js"
    assert_includes html, "navigator.doNotTrack"
    assert_includes html, token
  end

  def test_analytics_can_be_disabled_by_config
    build_site(
      env: "production",
      analytics: {
        "enabled" => false,
        "provider" => "cloudflare",
        "production_only" => true,
        "respect_dnt" => true,
        "cloudflare_token" => "test-token"
      }
    )

    html = rendered_post("interaction-essay")

    refute_includes html, "static.cloudflareinsights.com/beacon.min.js"
    refute_includes html, "data-site-analytics-loader"
  end

  def test_cloudflare_analytics_loads_only_in_production_with_token
    build_site(
      env: "production",
      analytics: {
        "enabled" => true,
        "provider" => "cloudflare",
        "production_only" => true,
        "respect_dnt" => true,
        "cloudflare_token" => "test-token"
      }
    )

    html = rendered_post("interaction-essay")

    assert_includes html, "data-site-analytics-loader"
    assert_includes html, "static.cloudflareinsights.com/beacon.min.js"
    assert_includes html, "navigator.doNotTrack"
    assert_includes html, "test-token"
  end

  def test_cloudflare_analytics_skips_development_builds
    build_site(
      env: "development",
      analytics: {
        "enabled" => true,
        "provider" => "cloudflare",
        "production_only" => true,
        "respect_dnt" => true,
        "cloudflare_token" => "test-token"
      }
    )

    html = rendered_post("interaction-essay")

    refute_includes html, "static.cloudflareinsights.com/beacon.min.js"
    refute_includes html, "data-site-analytics-loader"
  end

  def test_paired_post_embeds_english_body_and_enables_toggle
    build_site
    html = rendered_post("interaction-essay")

    assert_includes html, 'id="post-body-content"'
    assert_includes html, 'id="post-body-en"'
    assert_includes html, "data-pagefind-ignore"
    assert_includes html, "English body paragraph."
    assert_includes html, 'class="post-lang-toggle"'
    assert_includes html, 'data-lang="en"'
    refute_includes html, 'data-lang="en" disabled'
    assert_includes html, 'data-title-en="Interaction essay"'
    assert_includes html, 'data-intro-en="Interaction fixture."'
  end

  def test_unpaired_post_shows_disabled_english_toggle
    build_site
    html = rendered_post("interaction-solo") # unpaired zh-only essay

    assert_includes html, 'class="post-lang-toggle"'
    assert_includes html, 'title="暂无英文版"'
    refute_includes html, 'id="post-body-en"'
  end

  def test_paired_post_includes_toggle_runtime
    build_site
    html = rendered_post("interaction-essay")
    assert_includes html, "mf-lang"
    assert_includes html, "mf:langchange"
  end

  def test_toc_and_wordcount_target_visible_body
    build_site
    html = rendered_post("interaction-essay")
    assert_includes html, ".post-body:not([hidden])"
    assert_includes html, "mf:langchange"
  end

  def test_paired_post_emits_hreflang_alternates
    build_site
    html = rendered_post("interaction-essay")
    assert_includes html, 'hreflang="en"'
    assert_includes html, 'hreflang="zh-CN"'
  end

  def test_brief_index_excludes_english_variant
    build_site
    html = File.read(File.join(@dest_dir, "briefs", "index.html"))
    assert_equal 1, html.scan('"href":').length
    refute_includes html, ".en/"
  end

  private

  def write_post(filename, kind:, category:, body:)
    File.write(
      File.join(@site_dir, "_posts", filename),
      <<~MARKDOWN
        ---
        title: Interaction #{kind}
        date: 2026-05-18 12:00:00 +0800
        author: Test
        kind: #{kind}
        category: #{category}
        intro: Interaction fixture.
        ---

        #{body}
      MARKDOWN
    )
  end

  def build_site(env: "development", analytics: nil)
    previous_env = ENV["JEKYLL_ENV"]
    ENV["JEKYLL_ENV"] = env

    config_overrides = {
      "source" => @site_dir,
      "destination" => @dest_dir,
      "quiet" => true,
      "future" => true
    }
    config_overrides["analytics"] = analytics if analytics

    config = YAML.load_file(File.join(@site_dir, "_config.yml")).merge(config_overrides)
    Jekyll::Site.new(Jekyll.configuration(config)).process
  ensure
    ENV["JEKYLL_ENV"] = previous_env
  end

  def rendered_post(slug)
    File.read(File.join(@dest_dir, "notes", "2026", "05", "18", slug, "index.html"))
  end
end
