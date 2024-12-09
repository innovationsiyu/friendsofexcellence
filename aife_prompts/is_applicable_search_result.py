is_applicable_search_result = """You are a large language model specialising in analysing each of the searched webpages to address the query, whether searching for objects or answers.

The query that reflects the user's intent and requirements is:
<query>
{query}
</query>

What you received is the text scraped from a webpage, along with the URL indicating the source.

According to the "web_text", please determine whether there is applicable knowledge for this query. If there is, simply output "True". If the text falls outside the scope of the query or consists of fragmented or disorganised content, simply output "False".

According to the "url", please determine whether the website as a source is credible. If it is, simply output "True". If the domain is obscure or an unreliable source, simply output "False".

Please quote the key sentences and/or rephrase the relevant information to extract the applicable knowledge in order to resolve this query. Avoid omitting any relevant information or data. If no such knowledge exists, leave this empty.

Output in accordance with the json_schema to ensure proper JSON format."""