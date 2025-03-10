extract_structured_web_info = """The screenshot is a webpage. Does it contain the entire content of an article or other types of documents?

If you observe a complete article or document, identify and extract its title, source, published date, first paragraph and last paragraph, and output the structured info in accordance with the format example.

If no complete article or document is shown or if you are unable to observe the last paragraph of the body content, simply output "False".

Please be aware:
- The title should exclude any preceding and succeeding extra text and punctuation.
- The source should be the name of an institution or media outlet, rather than a media platform. If the page contains multiple sources, choose the one most likely to be the original, rather than the reposter.
- The published date should be in ISO format. If there are multiple dates, choose the one closest to the publication.
- The body content is the document's content from the author, including notes and appendices (if any), excluding any editorial notes, user comments, and similar additions.
- The first and last paragraphs mark the beginning text and end text of the body content.

Output format example:
<structured_info>
Title: 文章标题
Source: 文章来源
Published date: ISO格式的文章发表日期
First paragraph: 文章正文第一段的文本
Last paragraph: 文章正文最后一段的文本
</structured_info>

Output "False" if no complete article or document in the screenshot."""
