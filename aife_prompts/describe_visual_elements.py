describe_visual_elements = """Describe the visual elements of this page by addressing the following questions and tasks:

1. What charts, maps, diagrams, schematics, tables, and/or other specialised layouts are shown? Including:
   - Statistical and comparative: bar charts, line charts, pie charts, scatter plots, area charts, heat maps, box plots, histograms, bubble charts, radar charts, funnel charts, etc.
   - Relationship and flow: tree graphs, network graphs, Venn diagrams, tree maps, Sankey diagrams, chord diagrams, force-directed graphs, etc.
   - Temporal and spatial: time series plots, Gantt charts, candlestick charts, geographic thematic maps, flow maps, etc.
   - Schematics, tables, and any other visualisations of data and/or logic.

2. For each chart with labelled data, present each piece of data and its significance in a sequential manner.

3. For each chart without labelled data, explain the logic and conclusions you derive.

4. For each map, diagram, schematic, and other visualisation of data and/or logic, endeavour to explain all the logic and conclusions you derive from it.

5. For each table, endeavour to extract all cell contents in all rows and columns, preserving their precise original appearance. Avoid omitting any information or data from any cell.

If no such visual elements are shown, simply state that this page has no visual elements to describe.

Output in Markdown format using the same language as shown in the image."""
