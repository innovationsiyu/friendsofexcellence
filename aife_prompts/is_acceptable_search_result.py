is_acceptable_search_result = """The user is using search engines to retrieve information. Please assist in observing and analysing whether each searched webpage contains complete body content and aligns with the user's search target and acceptance criteria:

<search_target>
{search_target}
</search_target>

<acceptance_criteria>
{acceptance_criteria}
必须包含完整正文内容
必须是简体中文网页
不含有明显非理性的言论
不涉及批评政府或国家领导人的观点
</acceptance_criteria>

The screenshot is one of the searched webpages.

Does it contain the entire content of an article or other types of documents? If no complete article or document is shown or if you are unable to observe the last paragraph of the body content, simply output "False".

If you observe a complete article or document, does the content match the search target and meet the acceptance criteria? Review the full content and provide an analysis on its consistency with the search target and acceptance criteria.

If the content falls outside the scope of the search target or fails to comply with any points of the acceptance criteria, output "False" after the analysis."""
