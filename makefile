.PHONY: publish preview new setup-hugo install-hugo-apt install-mobile-export-deps export-mobile-image export-mobile-cards

all:

HUGO_MIN_VERSION := 0.146.0

MOBILE_SHOT_PORT := 13131
MOBILE_CARD_WIDTH := 430
MOBILE_CARD_HEIGHT := 932
# Use a 3x-class raster for sharper text on modern phone displays.
MOBILE_CARD_RENDER_DPI := 288
# 12.7 mm at 288 DPI, matching the page margin applied below.
MOBILE_CARD_TOP_SAFE_AREA := 144
# Applied to every PDF page before pagination.
MOBILE_CARD_TOP_MARGIN_MM := 12.7
SUDO ?= sudo

publish:
	@if [ -z "$(MSG)" ]; then echo "usage: make publish MSG="Example MSG" "; exit 1; fi
	git add .
	git commit -m "$(MSG)"
	git push

preview:
	hugo server -D --bind 0.0.0.0

setup-hugo:
	@echo "==> Checking Hugo environment"
	@if ! command -v hugo >/dev/null 2>&1; then \
		echo "error: hugo is not installed."; \
		echo "run 'make install-hugo-apt' on Debian/Ubuntu, then retry."; \
		exit 1; \
	fi
	@hugo version
	@if ! hugo version | grep -q '+extended'; then \
		echo "error: this site requires Hugo extended."; \
		exit 1; \
	fi
	@hugo version | grep -q 'v$(HUGO_MIN_VERSION)' || \
		echo "warning: expected Hugo >= $(HUGO_MIN_VERSION); please double-check your local version."
	@if [ ! -f ".gitmodules" ]; then \
		echo "warning: no .gitmodules found, skipping theme submodule setup."; \
	elif [ ! -d "themes/PaperMod/.git" ] && [ ! -f "themes/PaperMod/theme.toml" ]; then \
		echo "==> Initializing PaperMod submodule"; \
		git submodule update --init --recursive themes/PaperMod; \
	else \
		echo "==> PaperMod theme already present"; \
	fi
	@if [ ! -f "themes/PaperMod/theme.toml" ]; then \
		echo "error: themes/PaperMod is missing or incomplete."; \
		exit 1; \
	fi
	@echo "==> Hugo environment is ready"

install-hugo-apt:
	$(SUDO) apt update
	$(SUDO) apt install -y hugo
	@echo "==> Installed Hugo from apt"
	@hugo version

install-mobile-export-deps:
	$(SUDO) apt update
	$(SUDO) apt install -y chromium-browser fonts-noto-cjk poppler-utils wkhtmltopdf imagemagick
	@echo "==> Installed chromium-browser, fonts-noto-cjk, poppler-utils, wkhtmltopdf, and imagemagick"
	@command -v chromium-browser >/dev/null 2>&1 && chromium-browser --version || true
	@wkhtmltopdf --version


export:
	@if ! command -v wkhtmltopdf >/dev/null 2>&1; then echo 'error: wkhtmltopdf is not installed'; echo 'run: make install-mobile-export-deps'; exit 1; fi
	@if ! command -v pdftoppm >/dev/null 2>&1; then echo 'error: pdftoppm is not installed'; echo 'run: make install-mobile-export-deps'; exit 1; fi
	@if [ -n "$(PAGE)" ]; then \
		pages="$(PAGE)"; \
		batch=0; \
	else \
		pages=$$(hugo list all | awk -F, 'NR > 1 && $$9 == "page" && ($$10 == "technical-blog" || $$10 == "essays") { sub(/^https?:\/\/[^/]+/, "", $$8); print $$8 }'); \
		batch=1; \
	fi; \
	if [ -z "$$pages" ]; then echo 'error: no technical blog or essay pages were found'; exit 1; fi; \
	tmp_log=$$(mktemp); \
	tmp_pdf=$$(mktemp --suffix=.pdf); \
	trap 'if [ -n "$$server_pid" ]; then kill $$server_pid >/dev/null 2>&1 || true; wait $$server_pid >/dev/null 2>&1 || true; fi; rm -f "$$tmp_log" "$$tmp_pdf"' EXIT; \
	hugo server -D --bind 127.0.0.1 --port $(MOBILE_SHOT_PORT) >"$$tmp_log" 2>&1 & \
	server_pid=$$!; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		if grep -q 'Web Server is available at' "$$tmp_log"; then break; fi; \
		sleep 1; \
	done; \
	if ! grep -q 'Web Server is available at' "$$tmp_log"; then \
		cat "$$tmp_log"; \
		echo 'error: hugo server did not start in time'; \
		exit 1; \
	fi; \
	exported=0; \
	skipped=0; \
	for page in $$pages; do \
		page_slug=$$(printf '%s' "$$page" | sed 's#/*$$##; s#.*/##'); \
		if [ -z "$$page_slug" ]; then echo "error: invalid page path $$page"; exit 1; fi; \
		if [ "$$batch" -eq 1 ]; then \
			out_dir="exports/$$page_slug"; \
			set -- "$$out_dir"/page-*.png; \
			if [ -e "$$1" ]; then \
				echo "==> Skipping $$page (cards already exist in $$out_dir)"; \
				skipped=$$((skipped + 1)); \
				continue; \
			fi; \
		else \
			out_dir="$${OUT_DIR:-exports/$$page_slug}"; \
			rm -rf "$$out_dir"; \
		fi; \
		mkdir -p "$$out_dir"; \
		url="http://127.0.0.1:$(MOBILE_SHOT_PORT)$$page"; \
		case "$$url" in *\?*) url="$$url&theme=$${THEME:-dark}&export=mobile" ;; *) url="$$url?theme=$${THEME:-dark}&export=mobile" ;; esac; \
		rm -f "$$tmp_pdf"; \
		wkhtmltopdf --enable-local-file-access --background --javascript-delay 2000 --user-style-sheet "$$(pwd)/static/export-mobile.css" --page-width "$${WIDTH:-$(MOBILE_CARD_WIDTH)}px" --page-height "$${HEIGHT:-$(MOBILE_CARD_HEIGHT)}px" --margin-top $(MOBILE_CARD_TOP_MARGIN_MM)mm --margin-right 0 --margin-bottom 0 --margin-left 0 "$$url" "$$tmp_pdf" >/dev/null 2>&1; \
		if [ ! -s "$$tmp_pdf" ]; then echo "error: failed to render $$page"; exit 1; fi; \
		pdfinfo "$$tmp_pdf" >/dev/null 2>&1 || { echo "error: rendered PDF is invalid for $$page"; exit 1; }; \
		pdftoppm -png -rx $(MOBILE_CARD_RENDER_DPI) -ry $(MOBILE_CARD_RENDER_DPI) "$$tmp_pdf" "$$out_dir/page" >/dev/null; \
		set -- "$$out_dir"/page-*.png; \
		if [ ! -e "$$1" ]; then echo "error: no PNG pages were generated for $$page"; exit 1; fi; \
		if command -v mogrify >/dev/null 2>&1; then \
			card_geometry=$$(identify -format '%wx%h' "$$1"); \
			mogrify -trim +repage -background 'rgb(29, 30, 32)' -gravity north -splice 0x$(MOBILE_CARD_TOP_SAFE_AREA) -extent "$$card_geometry" "$$out_dir"/page-*.png; \
		fi; \
		echo "==> Exported $$page to $$out_dir"; \
		exported=$$((exported + 1)); \
	done; \
	echo "==> Exported $$exported article(s); skipped $$skipped article(s)"

new:
	@if [ -z "$(SECTION)" ] || [ -z "$(TITLE)" ]; then echo 'usage: make new SECTION="technical-blog|essays|publications" TITLE="Your Post Title"'; exit 1; fi
	@if [ ! -d "content/$(SECTION)" ]; then echo 'error: invalid SECTION "$(SECTION)"'; echo 'allowed sections: technical-blog, essays, publications'; exit 1; fi
	@slug=$$(printf '%s' "$(TITLE)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$$//'); \
	if [ -z "$$slug" ]; then echo 'error: TITLE produced an empty slug'; exit 1; fi; \
	if [ -e "content/$(SECTION)/$$slug.md" ]; then echo 'error: content/$(SECTION)/'$$slug'.md already exists'; echo 'tip: choose a different TITLE or rename/remove the existing file first'; exit 1; fi; \
	hugo new "$(SECTION)/$$slug.md"
