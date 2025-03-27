is_acceptable_search_result = """The user is using search engines to retrieve information. Please assist in observing and analysing whether the webpage as one of the search results contains a complete article or document and aligns with the user's search target and acceptance criteria:

<search_target>
{search_target}
</search_target>

<acceptance_criteria>
{acceptance_criteria}
文章或文件的完整正文文本在网页页面显示完全
文章或文件有明确的标题、发布机构、发布日期
不是广告或营销软文
没有批评政府或国家领导人的言论
</acceptance_criteria>

The text you received is the content scraped from a webpage as one of the search results.

Does it contain the entire content of an article or other types of documents? If no complete body text is shown or if you are unable to observe the end (last paragraph) of the body text, simply output "False".

If you observe a complete article or document, does the content match the search target and meet the acceptance criteria? Review the full content and provide an analysis of its consistency with the search target and acceptance criteria.

If the content falls outside the scope of the search target or fails to comply with the acceptance criteria, output "False" after the analysis."""
