.PHONY: publish preview new setup-hugo install-hugo-apt

all:

HUGO_MIN_VERSION := 0.146.0

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
	sudo apt update
	sudo apt install -y hugo
	@echo "==> Installed Hugo from apt"
	@hugo version

new:
	@if [ -z "$(SECTION)" ] || [ -z "$(TITLE)" ]; then echo 'usage: make new SECTION="technical-blog|essays|publications" TITLE="Your Post Title"'; exit 1; fi
	@if [ ! -d "content/$(SECTION)" ]; then echo 'error: invalid SECTION "$(SECTION)"'; echo 'allowed sections: technical-blog, essays, publications'; exit 1; fi
	@slug=$$(printf '%s' "$(TITLE)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$$//'); \
	if [ -z "$$slug" ]; then echo 'error: TITLE produced an empty slug'; exit 1; fi; \
	hugo new "$(SECTION)/$$slug.md"
