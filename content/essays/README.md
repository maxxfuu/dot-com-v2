# Blog Content

Write blog posts in this directory as flat markdown files:

- `content/blog/my-post.md`
- `content/blog/another-post.md`

Each post can use simple frontmatter like this:

```md
---
title: "My Post Title"
date: "2026-03-06"
summary: "One-line summary."
---
```

Posts are sorted by `date` in frontmatter (newest first) and published at `/blog/[slug]`.

## Images

Images and videos use normal markdown, with an optional size as the markdown title:

```md
![caption](/images/gemm/thing.png "small")
```

| size     | max height | max width | use for                                    |
| -------- | ---------- | --------- | ------------------------------------------ |
| `small`  | 16rem      | 50%       | square or portrait diagrams                |
| (none)   | 24rem      | 80%       | the default                                |
| `large`  | 32rem      | 95%       | dense diagrams that need the room          |
| `full`   | none       | 100%      | edge to edge of the reading column         |

Both caps apply and the tighter one wins, so aspect ratio is always preserved.
`small`, the default, and `large` never upscale past the file's own pixels;
`full` does stretch to the column, so only use it on an image wide enough to
take it (roughly 1500px or more to stay sharp on a high-DPI screen). Change the
numbers behind these names in `figureSizes` in
`app/(writings)/writings/components/blog-markdown.tsx` and every image follows.

Clicking any image opens it full size, so a diagram can be small in the page and
still readable.

## Other syntax

- ` ```cuda ` selects syntax highlighting; the language is never displayed.
  A bare ` ``` ` fence renders unhighlighted, which is what ASCII diagrams want.
- ` ```latex ` (or `tex` / `math`) typesets the fence as display math via KaTeX,
  rendered at build time so no JavaScript ships to the reader:

  ````md
  ```latex
  C_{ij} = \alpha \sum_{k=0}^{K-1} A_{ik} B_{kj} + \beta C_{ij}
  ```
  ````

  Bad LaTeX renders in red in place rather than failing the build.
- `<!-- ... -->` comments out a block, including across multiple lines.
- `[^1]` footnotes collect into a References section at the end of the page.
