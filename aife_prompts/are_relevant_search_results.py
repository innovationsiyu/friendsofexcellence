are_relevant_search_results = """The user is using search engines to retrieve information. Please assist in identifying which of the search results may be relevant to the user's search target:

<search_target>
{search_target}
</search_target>

The JSON data you received comprises some search results, each with an index.

Based on the "title" and "summary" of each item, please identify the ones that would be relevant to the search target and warrant a thorough reading for further inspection.

Output the indices of the relevant search results in a list format. Example: [0, 1, 8, 27]. If no relevant search results are found, output an empty list: [].

Avoid outputting anything else."""
