are_relevant_search_results = """The JSON data you received comprises some of the search results for the query:
<query>
{query}
</query>

According to the "title" and "summary" of each item, please determine which webpages are most likely to be relevant and warrant reading in full in order to resolve this query.

Output the indices of the relevant search results in a list format.

Output format example: [1, 3, 5]. If no relevant search results exist, output an empty list: [].

Avoid outputting anything else."""