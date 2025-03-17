describe_visual_elements = """Describe the visual elements of this page by addressing the following questions and tasks:

1. What chart elements, tables, schematics, mind maps, and/or other specialised layouts are shown? Describe each one that is shown, including:
   - Statistical and comparative: bar charts, line charts, pie charts, scatter plots, area charts, heat maps, box plots, histograms, bubble charts, radar charts, funnel charts, etc.
   - Relationship and flow: tree graphs, network graphs, Venn diagrams, tree maps, Sankey diagrams, chord diagrams, force-directed graphs, etc.
   - Temporal and spatial: time series plots, Gantt charts, candlestick charts, geographic thematic maps, flow maps, etc.
   - Tables, schematics, mind maps, and any other visualisations of data and/or logic.

2. For each chart element, describe all the information and insights you gathered from it:
   - For those with labelled data, present each piece of data and its significance in a sequential manner.
   - For those without labelled data, explain the logic and conclusions you derived.

3. For each table, endeavour to extract all cell contents in all rows and columns, preserving their precise original appearance. Avoid omitting any information or data from any cell.

4. For each schematic, mind map, and other visualisation of data and/or logic, endeavour to explain all the logic and conclusions you derived from it.

If no such visual elements are shown, simply state that this page has no visual elements to describe.

Output in Markdown format using the same language as shown in the image."""
