extract_structured_search_result = """The text you received is the content scraped from a webpage. Does it contain the entire content of an article or other types of documents?

If you observe a complete article or document, identify and extract its title, source, published date, first paragraph and last paragraph, and output the structured info in accordance with the format example.

If no complete article or document is shown or if you are unable to observe the end (last paragraph) of the body content, simply output "False".

Please be aware:
- The title should exclude any preceding and succeeding extra text and punctuation.
- The source should be the name of an institution or media outlet, rather than a social media platform. Choose the one most likely to be the original, rather than reposters, if you observe multiple sources.
- The published date should be the one closest to the publication if you observe multiple dates.
- The body content is the document's content from the author, including notes and appendices (if any), excluding any editorial notes, user comments, and similar additions.
- The first and last paragraphs mark the beginning text and end text of the body content.

Output format example:
<structured_info>
Title: 文章或文件的标题
Source: 文章或文件的发布机构
Published date: 文章或文件的发布日期
First paragraph: 文章或文件正文第一段的文本
Last paragraph: 文章或文件正文最后一段的文本
</structured_info>

Output "False" if no complete article or document in the screenshot."""
