.PHONY: publish preview new setup-hugo install-hugo-apt install-mobile-export-deps export-mobile-image export-mobile-cards

all:

HUGO_MIN_VERSION := 0.146.0
MOBILE_SHOT_PORT := 1313
MOBILE_SHOT_WIDTH := 430
MOBILE_SHOT_HEIGHT := 3200
MOBILE_CARD_WIDTH := 430
MOBILE_CARD_HEIGHT := 932
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

export-mobile-image:
	@if [ -z "$(PAGE)" ]; then echo 'usage: make export-mobile-image PAGE="/technical-blog/cute-starter-1/" OUT="exports/post.png"'; exit 1; fi
	@if ! command -v chromium-browser >/dev/null 2>&1; then echo 'error: chromium-browser is not installed'; echo 'run: make install-mobile-export-deps'; exit 1; fi
	@mkdir -p "$$(dirname "$${OUT:-exports/mobile-shot.png}")"
	@tmp_log=$$(mktemp); \
	trap 'if [ -n "$$server_pid" ]; then kill $$server_pid >/dev/null 2>&1 || true; wait $$server_pid >/dev/null 2>&1 || true; fi; rm -f "$$tmp_log"' EXIT; \
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
		url="http://127.0.0.1:$(MOBILE_SHOT_PORT)$${PAGE}"; \
		case "$$url" in *\?*) url="$$url&theme=$${THEME:-dark}" ;; *) url="$$url?theme=$${THEME:-dark}" ;; esac; \
		out_path="$${OUT:-exports/mobile-shot.png}"; \
		chromium-browser --headless --disable-gpu --hide-scrollbars --virtual-time-budget=4000 --window-size=$${WIDTH:-$(MOBILE_SHOT_WIDTH)},$${HEIGHT:-$(MOBILE_SHOT_HEIGHT)} --screenshot="$$out_path" "$$url"; \
		echo "==> Saved mobile image to $$out_path"

export-mobile-cards:
	@if [ -z "$(PAGE)" ]; then echo 'usage: make export-mobile-cards PAGE="/technical-blog/cute-starter-1/" OUT_DIR="exports/<page-slug>"'; exit 1; fi
	@if ! command -v wkhtmltopdf >/dev/null 2>&1; then echo 'error: wkhtmltopdf is not installed'; echo 'run: make install-mobile-export-deps'; exit 1; fi
	@if ! command -v pdftoppm >/dev/null 2>&1; then echo 'error: pdftoppm is not installed'; echo 'run: make install-mobile-export-deps'; exit 1; fi
	@page_slug=$$(printf '%s' "$${PAGE}" | sed 's#/*$$##; s#.*/##'); \
	if [ -z "$$page_slug" ]; then page_slug="mobile-cards"; fi; \
	out_dir="$${OUT_DIR:-exports/$$page_slug}"; \
	mkdir -p "$$out_dir"; \
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
	url="http://127.0.0.1:$(MOBILE_SHOT_PORT)$${PAGE}"; \
	case "$$url" in *\?*) url="$$url&theme=$${THEME:-dark}&export=mobile" ;; *) url="$$url?theme=$${THEME:-dark}&export=mobile" ;; esac; \
	wkhtmltopdf --enable-local-file-access --background --javascript-delay 2000 --user-style-sheet "$$(pwd)/static/export-mobile.css" --page-width "$${WIDTH:-$(MOBILE_CARD_WIDTH)}px" --page-height "$${HEIGHT:-$(MOBILE_CARD_HEIGHT)}px" --margin-top 0 --margin-right 0 --margin-bottom 0 --margin-left 0 "$$url" "$$tmp_pdf" >/dev/null 2>&1; \
	if [ ! -s "$$tmp_pdf" ]; then \
		echo 'error: failed to render PDF'; \
		exit 1; \
	fi; \
	pdfinfo "$$tmp_pdf" >/dev/null 2>&1 || { echo 'error: rendered PDF is invalid'; exit 1; }; \
	pdftoppm -png -rx 180 -ry 180 "$$tmp_pdf" "$$out_dir/page" >/dev/null; \
	if ! ls "$$out_dir"/page-*.png >/dev/null 2>&1; then \
		echo 'error: no PNG pages were generated'; \
		exit 1; \
	fi; \
	if command -v mogrify >/dev/null 2>&1; then \
		mogrify -trim +repage "$$out_dir"/page-*.png; \
	fi; \
	echo "==> Exported mobile cards to $$out_dir"

new:
	@if [ -z "$(SECTION)" ] || [ -z "$(TITLE)" ]; then echo 'usage: make new SECTION="technical-blog|essays|publications" TITLE="Your Post Title"'; exit 1; fi
	@if [ ! -d "content/$(SECTION)" ]; then echo 'error: invalid SECTION "$(SECTION)"'; echo 'allowed sections: technical-blog, essays, publications'; exit 1; fi
	@slug=$$(printf '%s' "$(TITLE)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$$//'); \
	if [ -z "$$slug" ]; then echo 'error: TITLE produced an empty slug'; exit 1; fi; \
	if [ -e "content/$(SECTION)/$$slug.md" ]; then echo 'error: content/$(SECTION)/'$$slug'.md already exists'; echo 'tip: choose a different TITLE or rename/remove the existing file first'; exit 1; fi; \
	hugo new "$(SECTION)/$$slug.md"
