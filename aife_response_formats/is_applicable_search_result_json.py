is_applicable_search_result_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "is_applicable_search_result",
        "schema": {
            "type": "object",
            "properties": {
                "is_applicable": {
                    "type": "boolean",
                    "description": "A True or False judgement on whether the text scraped from a webpage contains applicable knowledge for this query."
                },
                "is_credible": {
                    "type": "boolean",
                    "description": "A True or False judgement on whether the website as a source is credible."
                },
                "applicable_knowledge": {
                    "type": "string",
                    "description": "A knowledge extraction from the relevant information or data in order to resolve this query."
                }
            },
            "required": ["is_applicable", "is_credible", "applicable_knowledge"],
            "additionalProperties": False
        },
        "strict": True
    }
}
