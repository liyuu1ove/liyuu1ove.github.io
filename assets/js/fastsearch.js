const resList = document.getElementById('searchResults');
const sInput = document.getElementById('searchInput');
const searchBox = document.getElementById('searchbox');

let searchIndex = [];
let currentElement = null;
let firstResult = null;
let lastResult = null;

const debounce = (fn, delay) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = window.setTimeout(() => fn(...args), delay);
    };
};

const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const createMatcher = (query) => {
    const escapedQuery = escapeRegExp(query);
    const startsWithAsciiWord = /^[A-Za-z0-9_]/.test(query);
    const endsWithAsciiWord = /[A-Za-z0-9_]$/.test(query);
    const containsHan = /\p{Script=Han}/u.test(query);

    if (containsHan) {
        return new RegExp(`(${escapedQuery})`, 'iu');
    }

    const prefix = startsWithAsciiWord ? '(^|[^A-Za-z0-9_])' : '()';
    const suffix = endsWithAsciiWord ? '(?=$|[^A-Za-z0-9_])' : '';
    return new RegExp(`${prefix}(${escapedQuery})${suffix}`, 'iu');
};

const findMatch = (text, matcher) => {
    const value = String(text || '').replace(/\s+/g, ' ').trim();
    const match = matcher.exec(value);

    if (!match) {
        return null;
    }

    const prefixLength = match[1]?.length || 0;
    const matchedText = match[2] || match[1];
    const start = match.index + prefixLength;

    return { value, start, end: start + matchedText.length };
};

const createPreview = (match) => {
    const contextBefore = 72;
    const contextAfter = 120;
    let start = Math.max(0, match.start - contextBefore);
    let end = Math.min(match.value.length, match.end + contextAfter);

    if (start > 0) {
        const nextSpace = match.value.indexOf(' ', start);
        if (nextSpace !== -1 && nextSpace < match.start) {
            start = nextSpace + 1;
        }
    }

    if (end < match.value.length) {
        const previousSpace = match.value.lastIndexOf(' ', end);
        if (previousSpace > match.end) {
            end = previousSpace;
        }
    }

    const preview = document.createElement('p');
    preview.className = 'search-result-preview';
    preview.append(
        document.createTextNode(start > 0 ? '... ' : ''),
        document.createTextNode(match.value.slice(start, match.start))
    );

    const highlight = document.createElement('mark');
    highlight.textContent = match.value.slice(match.start, match.end);
    preview.append(highlight, document.createTextNode(match.value.slice(match.end, end)));

    if (end < match.value.length) {
        preview.append(document.createTextNode(' ...'));
    }

    return preview;
};

const normalizePermalink = (permalink) => {
    try {
        const url = new URL(permalink, window.location.href);
        return `${url.origin}${url.pathname.replace(/\/+$/, '') || '/'}${url.search}`;
    } catch {
        return String(permalink || '').replace(/\/+$/, '');
    }
};

const search = (query) => {
    const matcher = createMatcher(query);
    const seenPermalinks = new Set();
    const results = [];

    for (const item of searchIndex) {
        const titleMatch = findMatch(item.title, matcher);
        const contentMatch = findMatch(item.content, matcher);

        if (!titleMatch && !contentMatch) {
            continue;
        }

        const permalink = normalizePermalink(item.permalink);
        if (!permalink || seenPermalinks.has(permalink)) {
            continue;
        }

        seenPermalinks.add(permalink);
        results.push({ item, previewMatch: contentMatch || titleMatch, titleMatch: Boolean(titleMatch) });
    }

    return results.sort((left, right) => Number(right.titleMatch) - Number(left.titleMatch));
};

const reset = () => {
    currentElement = null;
    firstResult = null;
    lastResult = null;
    resList.innerHTML = '';
    sInput.value = '';
    sInput.focus();
};

const setActiveResult = (element) => {
    document.querySelectorAll('.focus').forEach((item) => item.classList.remove('focus'));

    if (!element) {
        return;
    }

    element.focus();
    element.parentElement?.classList.add('focus');
    currentElement = element;
};

const createResultIcon = () => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '24');
    svg.setAttribute('height', '24');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.classList.add('feather', 'feather-chevrons-right');
    svg.setAttribute('aria-hidden', 'true');
    svg.innerHTML = '<polyline points="13 17 18 12 13 7"></polyline><polyline points="6 17 11 12 6 7"></polyline>';
    return svg;
};

const renderResults = (results) => {
    if (!Array.isArray(results) || results.length === 0) {
        resList.innerHTML = '';
        firstResult = lastResult = currentElement = null;
        return;
    }

    const fragment = document.createDocumentFragment();

    for (const result of results) {
        const li = document.createElement('li');
        const copy = document.createElement('div');
        const title = document.createElement('div');
        const link = document.createElement('a');

        copy.className = 'search-result-copy';
        title.className = 'search-result-title';
        title.textContent = result.item.title;
        copy.append(title, createPreview(result.previewMatch));

        link.className = 'entry-link';
        link.href = result.item.permalink;
        link.setAttribute('aria-label', result.item.title);

        li.append(copy, createResultIcon(), link);
        fragment.appendChild(li);
    }

    resList.innerHTML = '';
    resList.appendChild(fragment);
    firstResult = resList.firstElementChild;
    lastResult = resList.lastElementChild;
};

const performSearch = () => {
    const query = sInput.value.trim();
    renderResults(query ? search(query) : []);
};

const initSearch = async () => {
    if (!sInput || !resList) {
        return;
    }

    sInput.disabled = false;
    sInput.focus();

    try {
        const response = await fetch('../index.json');
        if (!response.ok) {
            throw new Error(`Search index load failed: ${response.status}`);
        }

        const data = await response.json();
        searchIndex = Array.isArray(data) ? data : [];

        const query = new URLSearchParams(window.location.search).get('q');
        if (query) {
            sInput.value = query;
            performSearch();
        }
    } catch (error) {
        console.error(error);
    }
};

window.addEventListener('load', initSearch);
sInput?.addEventListener('input', debounce(performSearch, 150));

sInput?.addEventListener('search', () => {
    if (!sInput.value) {
        reset();
    }
});

document.addEventListener('keydown', (event) => {
    const { key } = event;
    const active = document.activeElement;
    const isInSearchBox = searchBox?.contains(active);

    if (key === 'Escape') {
        reset();
        return;
    }

    if (!firstResult || !isInSearchBox) {
        return;
    }

    if (key === 'ArrowDown') {
        event.preventDefault();

        if (active === sInput) {
            setActiveResult(firstResult.querySelector('.entry-link'));
        } else if (active?.parentElement !== lastResult) {
            setActiveResult(active?.parentElement?.nextElementSibling?.querySelector('.entry-link'));
        }
    } else if (key === 'ArrowUp') {
        event.preventDefault();

        if (active?.parentElement === firstResult) {
            setActiveResult(sInput);
        } else if (active !== sInput) {
            setActiveResult(active?.parentElement?.previousElementSibling?.querySelector('.entry-link'));
        }
    } else if (key === 'ArrowRight' && active?.matches?.('.entry-link')) {
        active.click();
    }
});
