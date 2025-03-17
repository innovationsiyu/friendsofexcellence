is_acceptable_search_result = """The user is using search engines to retrieve information. Please assist in observing and analysing whether each searched webpage contains complete body content and aligns with the user's search target and acceptance criteria:

<search_target>
{search_target}
</search_target>

<acceptance_criteria>
{acceptance_criteria}
文章或文件的语言是简体中文
文章或文件的完整正文在网页页面显示完全
文章或文件有明确的标题、发布机构、发布日期
没有批评中国政府或国家领导人的言论
</acceptance_criteria>

The text you received is the content scraped from one of the searched webpages.

Does it contain the entire content of an article or other types of documents? If no complete article or document is shown or if you are unable to observe the end (last paragraph) of the body content, simply output "False".

If you observe a complete article or document, does the content match the search target and meet the acceptance criteria? Review the full content and provide an analysis of its consistency with the search target and acceptance criteria.

If the content falls outside the scope of the search target or fails to comply with the acceptance criteria, output "False" after the analysis."""
