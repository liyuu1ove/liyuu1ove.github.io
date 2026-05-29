.PHONY: publish preview new

all:

publish:
	@if [ -z "$(MSG)" ]; then echo "usage: make publish MSG="Example MSG" "; exit 1; fi
	git add .
	git commit -m "$(MSG)"
	git push

preview:
	hugo server -D --bind 0.0.0.0

new:
	@if [ -z "$(SECTION)" ] || [ -z "$(TITLE)" ]; then echo 'usage: make new SECTION="technical-blog|essays|publications" TITLE="Your Post Title"'; exit 1; fi
	@if [ ! -d "content/$(SECTION)" ]; then echo 'error: invalid SECTION "$(SECTION)"'; echo 'allowed sections: technical-blog, essays, publications'; exit 1; fi
	@slug=$$(printf '%s' "$(TITLE)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$$//'); \
	if [ -z "$$slug" ]; then echo 'error: TITLE produced an empty slug'; exit 1; fi; \
	hugo new "$(SECTION)/$$slug.md"
