# Agent Guide for Technical Blog Work

## Scope

This document is for agents writing, editing, or reviewing technical blog posts in this repository.

The technical blog is not generic marketing content. It is a Chinese technical note series focused on GPU programming, high-performance computing, CUDA/CuTe/CUTLASS, distributed training, and collective communication. Changes should preserve technical correctness, local writing style, and Hugo compatibility.

Primary location:

- `content/technical-blog/`

Do not edit generated files in `public/` as source.

---

## Repository Context

This is a Hugo personal website using PaperMod.

Relevant stack:

- Hugo extended, expected version `0.146.0`
- PaperMod theme
- Markdown content with TOML front matter
- Syntax highlighting through Hugo/Chroma
- Local layout/CSS overrides outside the theme

Useful commands:

```bash
hugo -D
make preview
make new SECTION="technical-blog" TITLE="Post Title"
```

Use `hugo -D` when validating draft posts, because many technical blog posts are kept as drafts during editing.

---

## Technical Blog Structure

Technical blog posts live directly under:

```text
content/technical-blog/
```

Examples:

- `CuTe-starter-00.md`
- `CuTe-starter-0.md`
- `CuTe-starter-1.md`
- `CuTe-starter-2.md`
- `collective-communication-starter-0.md`
- `collective-communication-starter-1.md`

Post-specific images use page resource directories with the same basename as the post:

```text
content/technical-blog/CuTe-starter-2/
content/technical-blog/collective-communication-starter-1/
```

In the Markdown file, reference page resources by filename:

```markdown
![divide1](divide1.png)
```

Avoid paths like:

```markdown
![divide1](CuTe-starter-2/divide1.png)
```

Those can generate broken or duplicated paths depending on Hugo page bundle behavior.

---

## Front Matter

Technical blog front matter uses TOML:

```toml
+++
title = 'CuTe学习2-Layout Algebra'
date = '2026-06-25T16:35:03+08:00'
draft = false
description = '介绍定义在CuTe Layout的代数运算及其使用场景'
readingTimeText = '阅读此文大概需要31分钟'
tags = ['CuTe','Layout']
categories = ['Technical Blog']
+++
```

Rules:

- Preserve existing `title`, `date`, and `draft` unless explicitly asked to change them.
- Keep `categories = ['Technical Blog']`.
- Maintain manual `readingTimeText`; update it only when article length changes materially.
- Keep tag names stable within existing series where possible.
- Use single quotes to match existing posts.

---

## Current Technical Themes

### CuTe / CUTLASS Series

Current topics include:

- C++ template metaprogramming
- CuTe layout, tensor, layout algebra
- coalesce, composition, complement, logical divide, product
- tiling and thread/data mapping
- hierarchy tensor concepts
- CUDA kernel indexing and Tensor Core/MMA intuition

Future topics:
- CuTe GEMM walk through
- CUTLASS GEMM adoptor
- CUTLASS prelogue epilogue 
- costom CUTLASS collective, load for gemm-like kernel such as FlashAttention/GQA/MQA/MLA/DSA

Expected level:

- Explain from first principles.
- Connect abstractions to actual CUDA kernel concerns.
- Prefer concrete examples with small layouts over abstract prose alone.
- Use CuTe notation consistently:
  - `_3` for static integer
  - `3` for dynamic integer
  - `Shape:Stride` for Layout notation
  - nested parentheses for hierarchical tuples, e.g. `(2,(3,4))`
- Use CUTLASS/CuTe source code to explain how important classes are organized.
  - CUTLASS and CuTe source file is under `/home/azure/code/cutlass`, only use CUTLASS/CuTe source code reference in that folder.

### Collective Communication Series

Current topics:

- NCCL API and communicators
- GPU-initiated communication
- NVSHMEM
- NCCL GIN / Device API concepts
- collective primitives:
  - Broadcast
  - Gather
  - Reduce
  - Scatter
  - AllGather
  - AllReduce
  - ReduceScatter
  - AlltoAll

Future topics:
- Ring all reduce algorithm
- DP EP PP etc. training parallel tricks
- MoE All to All communication and optimization
- Megakernel

Expected level:

- Explain communication semantics with rank examples.
- Distinguish API surface from semantic primitive.
- Be careful with claims about new or emerging APIs such as NCCL GIN.
- When facts depend on current NVIDIA documentation or papers, verify against primary sources.

---

## Writing Style

Default language is Chinese.

Preferred style:

- Direct, explanatory, and technical.
- Start from the practical problem, then introduce abstraction.
- Use examples before heavy formalism when possible.
- Keep a first-person author voice where existing posts use it, e.g. `我们`, `作者`, `博主`.
- Casual technical phrasing is acceptable, but avoid making correctness depend on jokes or exaggeration.
- Keep terminology mixed Chinese/English when that is how the field uses it:
  - Layout
  - Tensor
  - Tile
  - Tiler
  - rank
  - communicator
  - stream
  - offset
  - stride
  - cosize
  - AllReduce

Avoid:

- Marketing tone.
- Overly polished textbook prose that removes the author's voice.
- Unexplained English paragraphs inside Chinese posts.
- Large claims without technical backing.
- Rewriting unrelated sections while fixing a local issue.

---

## Explanation Pattern

For difficult concepts, use this sequence:

1. State the practical reason the concept exists.
2. Define the concept in local notation.
3. Show a small numeric example.
4. Expand the mapping manually.
5. Explain what the result means in GPU/kernel terms.
6. Connect to higher-level APIs only after the low-level idea is clear.

Example for CuTe:

```text
thread layout -> data layout -> memory offset
```

Example for communication:

```text
local tensor on each rank -> collective primitive -> output tensor on each rank
```

---

## Markdown and Code Style

Use fenced code blocks with language tags:

```markdown
``` C++
auto layout = make_layout(make_shape(_4{}, _4{}),
                          make_stride(_4{}, _1{}));
```
```

Existing posts use both `C++` and `CPP`; prefer `C++` for new snippets unless surrounding code already uses `CPP`.

Use `plain text` for layout tables and mapping expansions:

```markdown
``` plain text
rank0: A0
rank1: A1
```
```

Use inline code for API names and notation:

- `composition(A, B)`
- `logical_divide(A, B)`
- `ncclAllReduce`
- `cosize`
- `Shape:Stride`

Do not use raw HTML unless Markdown cannot express the needed result. If image sizing or centering is required, prefer a repository-level CSS solution rather than scattered inline HTML.

---

## Images and Assets

Use page-specific image directories:

```text
content/technical-blog/<post-slug>/
```

Reference images by filename from the Markdown post:

```markdown
![AllReduce](allreduce.png)
```

After adding images, run:

```bash
hugo -D
```

Check generated HTML if images do not appear:

```bash
rg -n "<img|allreduce.png" public/technical-blog/<post-slug>/index.html
```

Common failure:

- Markdown uses `post-slug/image.png`
- Hugo copies resources beside `index.html`
- browser resolves the image as `post-slug/post-slug/image.png`
- result: broken image

Fix by using `image.png` directly.

---

## Technical Accuracy Rules

### For CuTe/CUTLASS

- Treat `Layout` as a mapping from logical coordinate to offset.
- Distinguish `size` and `cosize`.
- Distinguish logical shape from physical memory coverage.
- Do not imply operations are commutative unless proven.
- Composition order matters: `A o B` means `A(B(c))`.
- When discussing compile-time integers, be precise:
  - `Int<8>{}` is a static integer object/type-level value
  - `8` is runtime integer from CuTe's type perspective

### For NCCL/Distributed Training

- Distinguish rank, process, GPU, and communicator.
- Distinguish semantic collectives from NCCL's exact API surface.
- Do not claim NCCL has a standard one-call `AlltoAll` collective unless verified for the target version.
- For GIN / Device API / NCCL EP, avoid over-specific interface claims unless sourced from current primary docs or papers.
- Explain whether a communication model is:
  - host-initiated
  - GPU-initiated
  - collective
  - one-sided
  - point-to-point

---

## Source Policy

Use primary sources for technical claims when possible:

- NVIDIA CUTLASS/CuTe documentation
- NVIDIA NCCL documentation
- NVIDIA NVSHMEM documentation
- official papers or arXiv papers when the feature is research-stage

For current APIs, verify if there is any chance the information has changed.

Add references at the end of the post when introducing externally sourced examples or current systems:

```markdown
# Reference

https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html
```

Keep references concise. Do not paste long quoted passages from documentation.

---

## Editing Workflow

When asked to edit an existing technical blog:

1. Read the target section and surrounding context.
2. Preserve the author's existing terminology and notation.
3. Make the smallest change that fixes the issue unless asked to expand.
4. If adding examples, keep numbers small enough to verify manually.
5. Run `git diff --check` for the edited file.
6. Run `hugo -D` for Markdown/front matter/image changes.
7. Report generated `public/` changes as build artifacts, not source edits.

Recommended commands:

```bash
nl -ba content/technical-blog/<file>.md | sed -n 'START,ENDp'
git diff --check -- content/technical-blog/<file>.md
hugo -D
```

---

## Review Checklist

Before finalizing a technical blog edit, check:

- Front matter remains valid TOML.
- Headings are in the intended hierarchy.
- Code fences are closed.
- Image paths resolve from the generated page.
- No leftover placeholders such as `TODO`, bare `divide1.png`, or untranslated English explanatory paragraphs.
- Mathematical notation is internally consistent.
- The explanation connects back to the concrete GPU/distributed systems use case.
- The edit does not modify unrelated posts or generated output as source.

---

## Known Local Issues to Watch

- Some posts use page resource directories whose casing must match post slugs carefully.
- Hugo may copy page resources into generated `public/` output; do not edit those generated copies.
- `hugo -D` can refresh many files under `public/`; this is expected build output.
- Existing posts may contain drafts and partially written sections. Do not assume all placeholders are accidental unless they are in the requested edit scope.
- Existing prose intentionally mixes Chinese and English technical terms. Do not force full translation of API names or established terms.

---

## Minimal Rule Set

1. Keep technical content correct before making it elegant.
2. Preserve the local Chinese technical-note voice.
3. Use concrete examples for abstract concepts.
4. Keep image paths page-relative by filename.
5. Validate with `hugo -D` after content or asset changes.
6. Do not edit `public/` as source.
7. For current NVIDIA APIs or emerging systems, verify against primary sources before making strong claims.
