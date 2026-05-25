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
	@if [ -z "$(NEW_HUGO_PATH)" ]; then echo "usage: make new NEW_HUGO_PATH="path/to/file" "; exit 1; fi
	hugo new $(NEW_HUGO_PATH)