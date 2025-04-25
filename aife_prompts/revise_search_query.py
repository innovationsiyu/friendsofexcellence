revise_search_query = """The user is using search engines to retrieve information. Please assist in developing a new query for the user's search target and acceptance criteria:

<search_target>
{search_target}
</search_target>

<acceptance_criteria>
{acceptance_criteria}
</acceptance_criteria>

The JSON data you received comprises the search results and the query used.

Since the existing results are insufficient in quantity after filtering by the search target and acceptance criteria, please consider what further information should be added and define a more specific search target.

Generate a new query for further search targeting the information to be added. This can be keywords or phrases, and may also include advanced search operators.

Wrap the specific search target and new query within XML tags as specific_search_target and new_query when outputting:

<specific_search_target>
The more specific search target specifying the information to be added.
</specific_search_target>

<new_query>
The new query for further search targeting the information to be added. 
</new_query>
"""
